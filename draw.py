import os
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image as PilImage
import torch
from feeders.feeder import Feeder


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def draw_uncertain_attention_matrices(
    attentions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    path: str,
    aggregation: Optional[str] = None,
    cmap: str = "Greens",
    names: list[str] = ["Identity", "Inward", "Outward"],
    layers_to_draw: list[int] = [0, 4, 9],
):
    print(f"Drawing {path}")

    # Filter attentions to only include specified layers
    filtered_attentions = {k: v for k, v in attentions.items() if k in layers_to_draw}

    figure, axarr = plt.subplots(
        2 * 3,
        len(filtered_attentions.items()),
        figsize=(15 * 2 * 3, 15 * len(filtered_attentions.items())),
    )

    def aggregate(input: np.ndarray) -> np.ndarray:
        if aggregation == "mean":
            return input.mean(axis=0)
        if aggregation == "first":
            return input[0, :, :]
        if aggregation is None:
            return input
        raise ValueError()

    for i, (key, atts) in enumerate(filtered_attentions.items()):
        att, att_var = atts
        att_var = att_var / (att.abs() + 1e-7)
        att = att.detach().cpu().numpy().squeeze()
        att_var = att_var.detach().cpu().numpy().squeeze()

        for q in range(0, 3):
            ax = axarr[i, 0 + q * 2]
            data = aggregate(att[q])
            cax = ax.imshow(data, cmap=cmap)
            ax.set_title(f"Attention[{names[q]}] for layer {key}")
            cbar = figure.colorbar(cax, orientation="horizontal")

            ax = axarr[i, 1 + q * 2]
            data = aggregate(att_var[q])
            cax = ax.imshow(data, cmap=cmap)
            ax.set_title(f"Attention[{names[q]}] uncertainty for layer {key}")
            cbar = figure.colorbar(cax, orientation="horizontal")

    figure.savefig(path)


def crop_and_resize_by_skeletons(
    img: np.ndarray, 
    skeleton_coords_list: list[np.ndarray], 
    target_height: int = 1440, 
    padding: int = 100
) -> np.ndarray:
    all_x_coords = []
    all_y_coords = []

    for skeleton_coords in skeleton_coords_list:
        x_coords = [x for x, _, _ in skeleton_coords if x is not None]
        y_coords = [y for _, y, _ in skeleton_coords if y is not None]
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

    if not all_x_coords or not all_y_coords:
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))

    # Crop region calculation
    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    y_max = min(img.shape[0], y_max + padding)

    # Crop
    cropped = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]

    # Calculate width while maintaining aspect ratio
    aspect_ratio = cropped.shape[1] / cropped.shape[0]
    target_width = int(target_height * aspect_ratio)

    # Resize maintaining aspect ratio
    resized = cv2.resize(cropped, (target_width, target_height))

    return resized


def pad_images_to_max_width(all_images):
    # Find maximum width
    max_width = max(img.shape[1] for img in all_images)

    padded_images = []
    for img in all_images:
        height, width = img.shape[:2]
        padding_width = max_width - width
        padding = np.ones((height, padding_width, 3), dtype=np.uint8) * 255

        # Concatenate original image with padding
        padded_img = np.concatenate([img, padding], axis=1)
        padded_images.append(padded_img)

    return padded_images


def draw_uncertain_attention_skeleton_video_ntu60(
    attentions: list[dict],
    input_indexes: list[str],
    path: str,
    input_video_path="/data/sets/ntu3d/nturgb+d_rgb/",
    input_data_path="/data/sets/ntu3d/ntu60_unnromalized/xview/train_data_joint_2D+D.npy",
    input_labels_path="/data/sets/ntu3d/ntu60_unnromalized/xview/train_label.pkl",
    attention_layers_for_summary=[0, 4, 9],
    summary_frame=0,
    names=["Identity", "Inward", "Outward"],
):
    import pygifsicle
    import imageio

    dataset = create_ntu3d_dataset(input_data_path, input_labels_path)

    for i, single_attentions in zip(input_indexes, attentions):

        skeletons, label, index = dataset[i]
        sample_name_full = dataset.sample_name[i]
        sample_name = sample_name_full.replace(".skeleton", "")

        image_provider = VideoReader(
            os.path.join(input_video_path, sample_name + "_rgb.avi")
        )

        summary_figure, summary_axarr = plt.subplots(
            len(attention_layers_for_summary),
            3,
            figsize=(6 * len(attention_layers_for_summary), 6 * 3),
        )
        summary_pointer = 0

        poses_to_crop = None

        for attention_layer in single_attentions.keys():
            for attention_channel in range(0, 3):

                all_images = []
                img_prefix = path / f"videos/img"
                gif_name = (
                    path
                    / f"videos/{index}_att{attention_layer}_{attention_channel}_{sample_name}.gif"
                )

                for q, img in enumerate(image_provider):

                    frame_skeletons = skeletons[:, q, :, :]
                    transposed_frame_skeletons = [
                        frame_skeletons[:, :, i].transpose(1, 0)
                        for i in range(frame_skeletons.shape[-1])
                    ]
                    poses = [s for s in transposed_frame_skeletons if np.sum(s) != 0]

                    if poses_to_crop is None:
                        poses_to_crop = poses

                    img_uncertainty = img.copy()

                    att, att_var = single_attentions[attention_layer]
                    att_var = att_var / (att.abs() + 1e-7)
                    att = att.detach().cpu().numpy().squeeze()
                    att_var = att_var.detach().cpu().numpy().squeeze()

                    for pose in poses:
                        draw_uncertain_ntu3d_skeleton_2D(
                            img, img_uncertainty, pose, att, att_var, attention_channel
                        )

                    img = crop_and_resize_by_skeletons(img, poses_to_crop)
                    img_uncertainty = crop_and_resize_by_skeletons(
                        img_uncertainty, poses_to_crop
                    )

                    img = np.concatenate([img, img_uncertainty], axis=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    draw_image(
                        img,
                        f"{img_prefix}/{index}_att{attention_layer}_{attention_channel}_{sample_name}/{q}.png",
                    )

                    all_images.append(img)

                    if (
                        q == summary_frame
                        and attention_layer in attention_layers_for_summary
                    ):
                        ax = summary_axarr[summary_pointer, attention_channel]

                        ax.imshow(img)
                        ax.set_title(
                            f"Uncertain attention [{names[attention_channel]}] for layer {attention_layer}"
                        )

                        if attention_channel == 2:
                            summary_pointer += 1

                    print(f"[{i}]{sample_name} ({q+1})", end="\r")

                if len(all_images) > 0:

                    # all_images = pad_images_to_max_width(all_images)

                    summary_figure.savefig(path / f"summary_{index}_{sample_name}.png")
                    imageio.mimsave(gif_name, all_images)
                    pygifsicle.optimize(gif_name, options=["--no-conserve-memory"])

        print()


def draw_image(image, path, targets=[], colors=[]):

    if isinstance(image, np.ndarray):
        rgb_pi = image
    else:
        pi = image.mean(axis=0).detach().cpu().numpy()

        grayscale_pi = (
            (image.mean(axis=0) * 255 / image.mean(axis=0).max())
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        rgb_pi = np.stack([grayscale_pi] * 3, axis=-1)
        rgb_pi[pi < 0, 0] = 0

    for target, color in zip(targets, colors):

        pos_min = (target[0] - target[1] // 2).astype(np.int32)
        pos_max = (target[0] + (target[1] + 1) // 2 - 1).astype(np.int32)

        rgb_pi[pos_min[0] : pos_max[0] + 1, pos_min[1] : pos_max[1] + 1, :] = color

    os.makedirs(str(Path(path).parent), exist_ok=True)
    image = PilImage.fromarray(rgb_pi)
    image.save(path)

    return image


def draw_uncertain_ntu3d_skeleton_2D(
    img,
    img_uncertainty,
    pose,
    attention,
    attention_uncertainty,
    attention_channel=0,
    max_thickness=10,
    uncertainty_scale=0.5,
    color=np.array([0.0, 255.0, 255.0]),
    color_attention=np.array([255.0, 125.0, 55.0]),
    color_attention_uncertainty=np.array([0.0, 0.0, 255.0]),
):

    min_att = attention.min()
    max_att = attention.max()

    connecting_joint = [
        1,
        0,
        20,
        2,
        20,
        4,
        5,
        6,
        20,
        8,
        9,
        10,
        0,
        12,
        13,
        14,
        0,
        16,
        17,
        18,
        1,
        7,
        7,
        11,
        11,
    ]
    num_joints = 25

    # min_z = pose.data[:, 2].min()
    # max_z = pose.data[:, 2].max()

    for joint_a in range(num_joints):
        for joint_b in range(joint_a + 1, num_joints):

            x_a, y_a = pose[joint_a, :2]
            x_b, y_b = pose[joint_b, :2]

            exists_a = (x_a > 0) and (y_a > 0)
            exists_b = (x_b > 0) and (y_b > 0)

            if exists_a and exists_b:
                # thickness = int(
                #     uncertainty_scale * attention_uncertainty[attention_channel, joint_a, joint_b]
                # )
                # color = color_warm * temperature + color_cold * (1 - temperature)

                thickness = int(
                    uncertainty_scale
                    * max_thickness
                    * (
                        attention_uncertainty[attention_channel, joint_a, joint_b]
                        - min_att
                    )
                    / (max_att - min_att)
                )

                if thickness >= 1:
                    cv2.line(
                        img_uncertainty,
                        (int(x_a), int(y_a)),
                        (int(x_b), int(y_b)),
                        color_attention_uncertainty,
                        thickness,
                    )

    for joint_a in range(num_joints):
        for joint_b in range(joint_a + 1, num_joints):

            x_a, y_a = pose[joint_a, :2]
            x_b, y_b = pose[joint_b, :2]

            exists_a = (x_a > 0) and (y_a > 0)
            exists_b = (x_b > 0) and (y_b > 0)

            if exists_a and exists_b:
                thickness = int(
                    max_thickness
                    * (attention[attention_channel, joint_a, joint_b] - min_att)
                    / (max_att - min_att)
                )
                # color = color_warm * temperature + color_cold * (1 - temperature)

                if thickness > max_thickness:
                    thickness = max_thickness + 1

                if thickness >= 1:
                    cv2.line(
                        img,
                        (int(x_a), int(y_a)),
                        (int(x_b), int(y_b)),
                        color_attention,
                        thickness,
                    )

    for joint_a in range(num_joints):
        joint_b = connecting_joint[joint_a]
        x_a, y_a = pose[joint_a, :2]
        x_b, y_b = pose[joint_b, :2]

        exists_a = (x_a > 0) and (y_a > 0)
        exists_b = (x_b > 0) and (y_b > 0)

        if exists_a:
            # temperature = 0.0 # (z_a - min_z) / (max_z - min_z)
            # color = color_warm * temperature + color_cold * (1 - temperature)
            cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
        if exists_b:
            # temperature = 0.0 # (z_b - min_z) / (max_z - min_z)
            # color = color_warm * temperature + color_cold * (1 - temperature)
            cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
        if exists_a and exists_b:
            # temperature = 0.0 # ((z_b + z_a) / 2 - min_z) / (max_z - min_z)
            # color = color_warm * temperature + color_cold * (1 - temperature)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)


def draw_ntu3d_skeletons_3D(poses, path, additional_lines=None, skeleton_type="ntu3d"):
    from mpl_toolkits.mplot3d import Axes3D

    colors = ["b", "c", "m", "y"]

    connecting_joints = {
        "ntu3d": [
            [
                [1],
                [0],
                [20],
                [2],
                [20],
                [4],
                [5],
                [6],
                [20],
                [8],
                [9],
                [10],
                [0],
                [12],
                [13],
                [14],
                [0],
                [16],
                [17],
                [18],
                [1],
                [7],
                [7],
                [11],
                [11],
            ]
        ],
        "mediapipe": [
            [1, 4],
            [2],
            [3],
            [7],
            [5],
            [6],
            [8],
            [],
            [],
            [10],
            [],
            [12, 13, 23],
            [14, 24],
            [15],
            [16],
            [21, 17, 19],
            [22, 18, 20],
            [19],
            [20],
            [],
            [],
            [],
            [],
            [24, 25],
            [26],
            [27],
            [28],
            [29, 31],
            [30, 32],
            [31],
            [32],
            [],
            [],
        ],
    }[skeleton_type]

    num_joints = {
        "ntu3d": 25,
        "mediapipe": 33,
    }[skeleton_type]

    fig = plt.figure()
    plt.axis("scaled")
    for i, pose in enumerate(poses):

        color = colors[i % len(colors)]

        points = []
        lines = []

        for joint_a in range(num_joints):

            x_a, y_a, z_a = pose[joint_a][:3]
            exists_a = ((x_a != 0) and (y_a != 0)) or joint_a == 1
            if exists_a:
                points.append([(x_a, y_a, z_a), joint_a])

            for joint_b in connecting_joints[joint_a]:
                x_b, y_b, z_b = pose[joint_b][:3]

                exists_b = ((x_b != 0) and (y_b != 0)) or joint_a == 1

                # if exists_b:
                #     points.append((x_b, y_b, z_b))
                if exists_a and exists_b:
                    lines.append(
                        [
                            (x_a, y_a, z_a),
                            (x_b, y_b, z_b),
                        ]
                    )

        center = np.array([p[0] for p in points]).mean(axis=0)

        ax = fig.add_subplot(1, len(poses), i + 1, projection="3d")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        def plot_points_and_lines(points, lines, color):

            is_separate_colors = color is None

            if color is None:
                indices = [p[2] for p in points]
                color = [p[1] for p in points]
                points = [p[0] for p in points]
            else:
                indices = [p[1] for p in points]
                points = [p[0] for p in points]

            ax.scatter3D(
                [p[0] - center[0] for p in points],
                [p[1] - center[1] for p in points],
                [p[2] - center[2] for p in points],
                c=color,
            )

            # if not is_separate_colors:
            #     for i, p in zip(indices, points):
            #         ax.text(
            #             p[0] - center[0],
            #             p[1] - center[1],
            #             p[2] - center[2],
            #             str(i),
            #             None,
            #             size="small"
            #         )

            for line in lines:
                a, b = line[:2]

                if is_separate_colors:
                    c = line[2]
                else:
                    c = color

                ax.plot(
                    [a[0] - center[0], b[0] - center[0]],
                    [a[1] - center[1], b[1] - center[1]],
                    [a[2] - center[2], b[2] - center[2]],
                    c=c,
                )

        plot_points_and_lines(points, lines, color)
        points = []
        lines = []

        if additional_lines is not None:
            for q, (x, y, c) in enumerate(additional_lines[i]):
                points.append((x, -q, c))
                points.append((y, -q, c))
                lines.append([x, y, c])

        plot_points_and_lines(points, lines, None)

    os.makedirs(Path(path).parent, exist_ok=True)
    plt.savefig(path)


def create_ntu3d_dataset(
    data_path,
    labels_path,
    skeleton_data_type="joint",
):
    random_choose = False
    random_move = False
    window_size = -1

    result = Feeder(
        data_path=data_path,
        label_path=labels_path,
        random_choose=random_choose,
        random_move=random_move,
        window_size=window_size,
        # skeleton_data_type=skeleton_data_type,
        # data_name="nturgbd",
    )

    return result
