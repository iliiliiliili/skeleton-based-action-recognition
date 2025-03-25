from fire import Fire
from pathlib import Path
import os
import re
from dataclasses import dataclass
from typing import List
import json

DATASETS = ["ntu60", "ntu120", "kinetics"]
DATASET_SPLITS = {
    "ntu60": ["xview", "xsub"],
    "ntu120": ["xset", "xsub"],
    "kinetics": [""],
}
STREAM_TYPES = ["joint", "joint_bone"]
MODEL_TYPES = ["baselines", "vnn"]
BASELINE_MODELS = ["agcn", "stgcn"]
ST3D_MODELS = ["st3dgcn", "st3dgcnt"]
VNN_MODELS = ["vagcn", "vstgcn"]
UA_VNN_MODELS = ["uaeavagcn", "uaeavstgcn"]
VNN_DEFAULT_TRAINING_SAMPLES = 2
VNN_TRAINING_SAMPLES = [1, 2, 3, 4, 6, 8]
NTU_CLASSES = [60, 120]

datasets = {
    "kinetics": {
        "root": "/data/sets/kinetics",
        "joint": {
            "train": {
                "data_path": "train_data_joint.npy",
                "label_path": "train_label.pkl",
            },
            "test": {
                "data_path": "val_data_joint.npy",
                "label_path": "val_label.pkl",
            },
        },
        "joint_bone": {
            "train": {
                "data_path": "train_data_joint_bone.npy",
                "label_path": "train_label.pkl",
            },
            "test": {
                "data_path": "val_data_joint_bone.npy",
                "label_path": "val_label.pkl",
            },
        },
    },
    "ntu": {
        "root": lambda classes_count, split: f"/data/sets/ntu3d/ntu{classes_count}/{split}",
        "joint": {
            "train": {
                "data_path": "train_data_joint.npy",
                "label_path": "train_label.pkl",
            },
            "test": {
                "data_path": "val_data_joint.npy",
                "label_path": "val_label.pkl",
            },
        },
        "joint_bone": {
            "train": {
                "data_path": "train_data_joint_bone.npy",
                "label_path": "train_label.pkl",
            },
            "test": {
                "data_path": "val_data_joint_bone.npy",
                "label_path": "val_label.pkl",
            },
        },
    },
}


def create_yaml(data: list, path: str, spaces: int = 4):

    def create_text(data: list, indentation: int):
        result = ""

        for d in data:
            if len(d) == 0:
                result += "\n"
            elif len(d) == 1:
                result += " " * spaces * indentation + f"{d[0]}\n"
            else:
                if isinstance(d[1], list):
                    result += " " * spaces * indentation + f"{d[0]}:\n"
                    result += create_text(d[1], indentation + 1)
                else:
                    result += " " * spaces * indentation + f"{d[0]}: {d[1]}\n"

        return result

    text = create_text(data, 0)

    with open(path, "w") as f:
        print(text, file=f, end="")


def create_root_configs(path):

    os.makedirs(str(path), exist_ok=True)

    create_yaml(
        [
            ["device", "-1 # use all available GPUs"],
            [],
            ["nesterov", True],
            ["end_test", True],
        ],
        path / "base.yaml",
    )

    create_yaml(
        [
            [
                "model_args",
                [
                    ["samples", 1],
                    ["test_samples", 2],
                ],
            ],
            [],
            ["test_samples", "[1,2,3,4,5,10,15,20,30,40]"],
            ["test_batch_sizes", "[512,512,512,512,512,256,256,128,128,128]"],
        ],
        path / "vnn.yaml",
    )

    print("Created root configs")


def create_agcn_stgcn_configs(path, model_name):

    os.makedirs(str(path), exist_ok=True)

    model = {
        "agcn": "model.agcn.AGCN",
        "stgcn": "model.stgcn.STGCN",
    }[model_name]

    base_params = [
        ["work_dir", "./runs/baselines/$DATASET/$SPLIT/$STREAMS/$MODEL"],
        [],
        ["model", model],
        [],
        ["MODEL_NAME", model_name],
        [],
    ]

    includes = (["include", [["- base"]]],)

    create_yaml(
        [
            *base_params,
            *includes,
        ],
        path / "train.yaml",
    )

    create_yaml(
        [
            *base_params,
            ["weights", "./runs/baselines/$DATASET/$SPLIT/$STREAMS/$MODEL.best.pt"],
            ["phase", "test"],
            [],
            *includes,
        ],
        path / "test.yaml",
    )


def create_st3d_configs(path, model_name):

    os.makedirs(str(path), exist_ok=True)

    model = {
        "st3dgcn": "model.st3dgcn.ST3DGCN",
        "st3dgcnt": "model.st3dgcnt.ST3DGCNT",
    }[model_name]

    base_params = lambda temporal_kernel_size, temporal_stride=1: [
        ["work_dir", f"./runs/baselines/$DATASET/$SPLIT/$STREAMS/$MODEL_tks{temporal_kernel_size}_ts{temporal_stride}"],
        [],
        ["model", model],
        [],
        ["MODEL_NAME", model_name],
        [],
    ]

    includes = (["include", [["- base"]]],)

    for temporal_kernel_size in [3, 5, 7, 11]:
        create_yaml(
            [
                *base_params(temporal_kernel_size),
                ["model_args", [["temporal_kernel_size", temporal_kernel_size], ["temporal_padding", (temporal_kernel_size - 1) // 2]]],
                *includes,
            ],
            path / f"train_tks{temporal_kernel_size}.yaml",
        )
        
        create_yaml(
            [
                *base_params(temporal_kernel_size),
                ["model_args", [["temporal_kernel_size", temporal_kernel_size], ["temporal_padding", (temporal_kernel_size - 1) // 2]]],
                ["weights", f"./runs/baselines/$DATASET/$SPLIT/$STREAMS/$MODEL_tks{temporal_kernel_size}.best.pt"],
                ["phase", "test"],
                [],
                *includes,
            ],
            path / f"test_tks{temporal_kernel_size}.yaml",
        )

    for temporal_kernel_size, temporal_stride in [
        (3, 2), 
        (5, 2),
        (10, 2),
        (15, 2),
        (20, 2),
        (30, 2),
        (3, 3), 
        (5, 3),
        (10, 3),
        (15, 3),
        (20, 3),
    ]:
        create_yaml(
            [
                *base_params(temporal_kernel_size, temporal_stride),
                ["model_args", [["temporal_kernel_size", temporal_kernel_size], ["temporal_stride", temporal_stride]]],
                *includes,
            ],
            path / f"train_tks{temporal_kernel_size}_ts{temporal_stride}.yaml",
        )
        
        create_yaml(
            [
                *base_params(temporal_kernel_size, temporal_stride),
                ["model_args", [["temporal_kernel_size", temporal_kernel_size], ["temporal_stride", temporal_stride]]],
                ["weights", f"./runs/baselines/$DATASET/$SPLIT/$STREAMS/$MODEL_tks{temporal_kernel_size}_ts{temporal_stride}.best.pt"],
                ["phase", "test"],
                [],
                *includes,
            ],
            path / f"test_tks{temporal_kernel_size}_ts{temporal_stride}.yaml",
        )


def create_vnn_agcn_stgcn_configs(path, model_name, dataset):

    os.makedirs(str(path), exist_ok=True)

    model = {
        "vagcn": "model.vnn_agcn.VAGCN",
        "vstgcn": "model.vnn_stgcn.VStgcn",
    }[model_name]

    baseline_model_name = {
        "vagcn": "agcn",
        "vstgcn": "stgcn",
    }[model_name]

    base_params = lambda samples, name_suffix="", model_params=[]: [
        [
            "work_dir",
            f"./runs/vnn/$DATASET/$SPLIT/$STREAMS/$MODEL_s$SAMPLES_b$BATCH_SIZE{name_suffix}",
        ],
        [],
        ["model", model],
        [
            "model_args",
            [
                ["samples", samples],
                *model_params,
            ],
        ],
        [],
        ["MODEL_NAME", model_name],
        [],
    ]

    includes = (["include", [["- base"], ["- /vnn"]]],)
    includes_longer = (["include", [[f"- /{dataset}/longer"], ["- base"], ["- /vnn"]]],)

    def create_train_test(
        name_suffix,
        params,
        local_includes=includes,
        model_name_suffix=None,
        weights=lambda model_name_suffix: f"./runs/vnn/$DATASET/$SPLIT/$STREAMS/$MODEL_s$SAMPLES_b$BATCH_SIZE{model_name_suffix}.best.pt",
    ):
        create_yaml(
            [
                *params,
                *local_includes,
            ],
            path / f"train{name_suffix}.yaml",
        )

        create_yaml(
            [
                *params,
                ["weights", weights(name_suffix if model_name_suffix is None else model_name_suffix)],
                ["phase", "test"],
                [],
                *includes,
            ],
            path / f"test{name_suffix}.yaml",
        )

    create_train_test("", base_params(VNN_DEFAULT_TRAINING_SAMPLES))

    def create_iv_configs(training_samples):

        for init_vnn_name, init_vnn_weights in [
            ("usual", "usual"),
            ("f0x3", "fill:stds:0.001:0.001"),
            ("f0x4", "fill:stds:0.0001:0.0001"),
            ("xu0b0x2", "xavier_uniform0b:stds:0.01:0.001"),
            ("xufb0x2", "xavier_uniform_fb:stds:0.01:0.001"),
            ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
            ("xn0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
        ]:

            create_train_test(
                f"_iv_{init_vnn_name}_s{training_samples}",
                [
                    *base_params(training_samples, f"_iv_{init_vnn_name}", [
                        ["INIT_WEIGHTS", init_vnn_weights]
                    ]),
                    ["init_vnn_from", f"./runs/baselines/$DATASET/$SPLIT/$STREAMS/{baseline_model_name}.best.pt"],
                    [],
                ],
                model_name_suffix=f"_iv_{init_vnn_name}",
            )


    for training_samples in VNN_TRAINING_SAMPLES:
        create_train_test(f"_s{training_samples}", base_params(training_samples))

        create_train_test(
            f"_longer_s{training_samples}",
            base_params(training_samples, "_longer"),
            includes_longer,
            model_name_suffix="_longer",
        )

        create_train_test(
            f"_slr_s{training_samples}",
            [
                *base_params(training_samples, "_slr"),
                ["base_lr", 0.05],
                [],
            ],
            model_name_suffix="_slr",
        )

        create_train_test(
            f"_slrhb_s{training_samples * 2}",
            [
                *base_params(training_samples * 2, "_slr"),
                ["base_lr", 0.05],
                ["batch_size", 32],
                [],
            ],
            model_name_suffix="_slr",
        )

        create_train_test(
            f"_bpb_s{training_samples}",
            [
                *base_params(training_samples, "_bpb"),
                ["batches_per_backpropagation", 4],
                [],
            ],
            model_name_suffix="_bpb",
        )

        create_iv_configs(training_samples)


def create_uavnn_agcn_stgcn_configs(path, model_name, dataset):

    os.makedirs(str(path), exist_ok=True)

    model = {
        "uaeavagcn": "model.vnn_multioutput_agcn.UncertaintyAwareEarlyAttentionVAGCN",
        "uaeavstgcn": "model.vnn_multioutput_stgcn.UncertaintyAwareEarlyAttentionSTGCN",
    }[model_name]

    baseline_model_name = {
        "uaeavagcn": "agcn",
        "uaeavstgcn": "stgcn",
    }[model_name]

    base_params = lambda samples, name_suffix="", model_params=[]: lambda attention_filter_limit=None, training_method=None: [
        [
            "work_dir",
            f"./runs/vnn/$DATASET/$SPLIT/$STREAMS/$MODEL_s$SAMPLES_b$BATCH_SIZE{name_suffix}_{training_method}_trained"
            + (f"_afl{attention_filter_limit}" if attention_filter_limit is not None else ""),  
        ],
        [],
        ["model", model],
        [
            "model_args",
            [
                ["samples", samples],
                *([["attention_filter_limit", attention_filter_limit]] if attention_filter_limit is not None else []),
                *([["training_method", training_method]] if training_method is not None else []),
                *model_params,
            ],
        ],
        [],
        ["MODEL_NAME", model_name],
        [],
    ]

    includes = (["include", [["- base"], ["- /vnn"]]],)
    includes_longer = (["include", [[f"- /{dataset}/longer"], ["- base"], ["- /vnn"]]],)

    attention_filter_limits = [0.1, 0.2, 0.5, 0.7, 1.0]

    def create_train_test(
        name_suffix,
        params,
        local_includes=includes,
        model_name_suffix=None,
        weights=lambda model_name_suffix: f"./runs/vnn/$DATASET/$SPLIT/$STREAMS/$MODEL_s$SAMPLES_b$BATCH_SIZE{model_name_suffix}.best.pt",
    ):
        
        for training_method in ["variational", "uncertainty_aware"]:
            create_yaml(
                [
                    *params(None, training_method),
                    *local_includes,
                ],
                path / f"train_{training_method}{name_suffix}.yaml",
            )

            for afl in attention_filter_limits:
                create_yaml(
                    [
                        *params(afl, training_method),
                        ["weights", weights((name_suffix if model_name_suffix is None else model_name_suffix) + "_" + training_method + "_trained")],
                        ["phase", "test"],
                        [],
                        *includes,
                    ],
                    path / f"test_{training_method}_trained{name_suffix}_afl{afl}.yaml",
                )

    create_train_test("", base_params(VNN_DEFAULT_TRAINING_SAMPLES))

    def create_iv_configs(training_samples):

        for init_vnn_name, init_vnn_weights in [
            ("usual", "usual"),
            ("f0x3", "fill:stds:0.001:0.001"),
            ("f0x4", "fill:stds:0.0001:0.0001"),
            ("xu0b0x2", "xavier_uniform0b:stds:0.01:0.001"),
            ("xufb0x2", "xavier_uniform_fb:stds:0.01:0.001"),
            ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
            ("xn0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
        ]:

            create_train_test(
                f"_iv_{init_vnn_name}_s{training_samples}",
                lambda afl, training_method: [
                    *base_params(training_samples, f"_iv_{init_vnn_name}", [
                        ["INIT_WEIGHTS", init_vnn_weights]
                    ])(afl, training_method),
                    ["init_vnn_from", f"./runs/baselines/$DATASET/$SPLIT/$STREAMS/{baseline_model_name}.best.pt"],
                    [],
                ],
                model_name_suffix=f"_iv_{init_vnn_name}",
            )


    for training_samples in VNN_TRAINING_SAMPLES:
        create_train_test(f"_s{training_samples}", base_params(training_samples))
        create_iv_configs(training_samples)


def create_kinetics_configs(path):

    os.makedirs(str(path), exist_ok=True)

    create_yaml(
        [
            [
                "model_args",
                [
                    ["num_class", 400],
                    ["num_point", 18],
                    ["num_person", 2],
                    ["graph", "graph.kinetics.Graph"],
                    ["graph_args", [["labeling_mode", "'spatial'"]]],
                ],
            ],
            [],
            ["#optimization"],
            ["weight_decay", 0.0001],
            ["base_lr", 0.1],
            ["step", "[45, 55]"],
            [],
            ["batch_size", 64],
            ["test_batch_size", 64],
            ["num_epoch", 65],
            [],
            ["DATASET_NAME", "kinetics"],
            ["SPLIT_NAME", "all"],
            [],
            ["include", [["- base"]]],
        ],
        path / "base.yaml",
    )
    
    create_yaml(
        [
            ["#optimization"],
            ["step", "[55, 70]"],
            ["num_epoch", 80],
        ],
        path / "longer.yaml",
    )

    for stream_type in STREAM_TYPES:
        stream_path = path / stream_type

        os.makedirs(str(stream_path), exist_ok=True)

        create_yaml(
            [
                ["feeder", "feeders.feeder.Feeder"],
                [
                    "train_feeder_args",
                    [
                        [
                            "data_path",
                            datasets["kinetics"]["root"]
                            + "/"
                            + datasets["kinetics"][stream_type]["train"]["data_path"],
                        ],
                        [
                            "label_path",
                            datasets["kinetics"]["root"]
                            + "/"
                            + datasets["kinetics"][stream_type]["train"]["label_path"],
                        ],
                        ["debug", False],
                        ["random_choose", False],
                        ["random_shift", False],
                        ["random_move", False],
                        ["window_size", -1],
                        ["normalization", False],
                    ],
                ],
                [],
                [
                    "test_feeder_args",
                    [
                        [
                            "data_path",
                            datasets["kinetics"]["root"]
                            + "/"
                            + datasets["kinetics"][stream_type]["test"]["data_path"],
                        ],
                        [
                            "label_path",
                            datasets["kinetics"]["root"]
                            + "/"
                            + datasets["kinetics"][stream_type]["test"]["label_path"],
                        ],
                        ["debug", False],
                    ],
                ],
                [],
                ["STREAMS_NAME", stream_type],
                ([] if stream_type == "joint" else ["model_args", [["in_channels", 6]]]),
                [],
                ["include", [["- base"]]],
            ],
            stream_path / "base.yaml",
        )

        for model_name in BASELINE_MODELS:
            create_agcn_stgcn_configs(stream_path / model_name, model_name)

        for model_name in ST3D_MODELS:
            create_st3d_configs(stream_path / model_name, model_name)

        for model_name in VNN_MODELS:
            create_vnn_agcn_stgcn_configs(stream_path / model_name, model_name, f"kinetics")

        for model_name in UA_VNN_MODELS:
            create_uavnn_agcn_stgcn_configs(stream_path / model_name, model_name, f"kinetics")

    print("Created kinetics configs")


def create_ntu_configs(path, classes_count):

    os.makedirs(str(path), exist_ok=True)

    create_yaml(
        [
            [
                "model_args",
                [
                    ["num_class", classes_count],
                    ["num_point", 25],
                    ["num_person", 2],
                    ["graph", "graph.ntu_rgb_d.Graph"],
                    ["graph_args", [["labeling_mode", "'spatial'"]]],
                ],
            ],
            [],
            ["#optimization"],
            ["weight_decay", 0.0001],
            ["base_lr", 0.1],
            ["step", "[30, 40]"],
            [],
            ["batch_size", 64],
            ["test_batch_size", 64],
            ["num_epoch", 50],
            [],
            ["DATASET_NAME", f"ntu{classes_count}"],
            [],
            ["include", [["- base"]]],
        ],
        path / "base.yaml",
    )
    
    create_yaml(
        [
            ["#optimization"],
            ["step", "[40, 55]"],
            ["num_epoch", 65],
        ],
        path / "longer.yaml",
    )

    for split in DATASET_SPLITS[f"ntu{classes_count}"]:
        split_path = path / split

        os.makedirs(str(split_path), exist_ok=True)

        create_yaml(
            [
                ["SPLIT_NAME", split],
                [],
                ["include", [["- base"]]],
            ],
            split_path / "base.yaml",
        )

        for stream_type in STREAM_TYPES:
            stream_path = split_path / stream_type

            os.makedirs(str(stream_path), exist_ok=True)

            create_yaml(
                [
                    ["feeder", "feeders.feeder.Feeder"],
                    [
                        "train_feeder_args",
                        [
                            [
                                "data_path",
                                datasets["ntu"]["root"](classes_count, split)
                                + "/"
                                + datasets["ntu"][stream_type]["train"]["data_path"],
                            ],
                            [
                                "label_path",
                                datasets["ntu"]["root"](classes_count, split)
                                + "/"
                                + datasets["ntu"][stream_type]["train"]["label_path"],
                            ],
                            ["debug", False],
                            ["random_choose", False],
                            ["random_shift", False],
                            ["random_move", False],
                            ["window_size", -1],
                            ["normalization", False],
                        ],
                    ],
                    [],
                    [
                        "test_feeder_args",
                        [
                            [
                                "data_path",
                                datasets["ntu"]["root"](classes_count, split)
                                + "/"
                                + datasets["ntu"][stream_type]["test"]["data_path"],
                            ],
                            [
                                "label_path",
                                datasets["ntu"]["root"](classes_count, split)
                                + "/"
                                + datasets["ntu"][stream_type]["test"]["label_path"],
                            ],
                            ["debug", False],
                        ],
                    ],
                    [],
                    ["STREAMS_NAME", stream_type],
                    ([] if stream_type == "joint" else ["model_args", [["in_channels", 6]]]),
                    [],
                    ["include", [["- base"]]],
                ],
                stream_path / "base.yaml",
            )

            for model_name in BASELINE_MODELS:
                create_agcn_stgcn_configs(stream_path / model_name, model_name)
                
            for model_name in ST3D_MODELS:
                create_st3d_configs(stream_path / model_name, model_name)

            for model_name in VNN_MODELS:
                create_vnn_agcn_stgcn_configs(stream_path / model_name, model_name, f"ntu{classes_count}")
                
            for model_name in UA_VNN_MODELS:
                create_uavnn_agcn_stgcn_configs(stream_path / model_name, model_name, f"ntu{classes_count}")

    print(f"Created ntu{classes_count} configs")


def main(root="./config"):

    root_path = Path(root)

    create_root_configs(root_path)
    create_kinetics_configs(root_path / "kinetics")

    for classes_count in NTU_CLASSES:
        create_ntu_configs(root_path / f"ntu{classes_count}", classes_count)


if __name__ == "__main__":
    Fire(main)
