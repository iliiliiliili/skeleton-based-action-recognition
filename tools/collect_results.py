from fire import Fire
import os
import re
from dataclasses import dataclass
from typing import List
import json
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    scale_x_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    theme,
    element_text,
    ylab,
    xlab,
    scale_color_discrete,
    labeller,
)
from tabulate import tabulate, SEPARATING_LINE
import datetime
import simple_colors as colors

DATASET_FLAGS = ["ntu60", "ntu120", "kinetics"]
SPLIT_FLAGS = ["xset", "xview", "xsub", "all"]
DATASET_SPLIT_FLAGS = {
    "ntu60": ["xview", "xsub"],
    "ntu120": ["xset", "xsub"],
    "kinetics": ["all"],
}
SKELETON_TYPE_FLAGS = ["joint", "joint_bone"]
MODEL_TYPE_FLAGS = ["baselines", "vnn"]


@dataclass
class SingleResult:
    samples: int
    batch: int
    top1: float
    top5: float


@dataclass
class Experiment:
    network_type: str
    samples: int
    batch: int
    flags: List[str]
    results: List[SingleResult]
    age_days: int

    def best_result(self):
        return max([r for r in self.results], key=lambda r: r.top1)

    def __str__(self):
        result = f"Experiment(network_type={self.network_type}, samples={self.samples}, batch={self.batch}, age_days={self.age_days}, flags={self.flags}\n"

        for r in self.results:
            result += f"    {r}\n"

        result += ")"
        return result


def file_age_in_days(path):
    return (
        datetime.datetime.today()
        - datetime.datetime.fromtimestamp(os.path.getmtime(path))
    ).days


def group_experiments(experiments: List[Experiment]):

    result = {}

    for dataset in DATASET_FLAGS:
        for split in SPLIT_FLAGS:
            for skeleton_type in SKELETON_TYPE_FLAGS:
                for model_type in MODEL_TYPE_FLAGS:
                    if dataset not in result:
                        result[dataset] = {}
                    if split not in result[dataset]:
                        result[dataset][split] = {}
                    if skeleton_type not in result[dataset][split]:
                        result[dataset][split][skeleton_type] = {}
                    if model_type not in result[dataset][split][skeleton_type]:
                        result[dataset][split][skeleton_type][model_type] = []

    for experiment in experiments:
        dataset = None
        split = ""
        skeleton_type = None
        model_type = None

        for flag in experiment.flags:
            if flag in DATASET_FLAGS:
                dataset = flag
            elif flag in SPLIT_FLAGS:
                split = flag
            elif flag in SKELETON_TYPE_FLAGS:
                skeleton_type = flag
            elif flag in MODEL_TYPE_FLAGS:
                model_type = flag

        if (
            (dataset is not None)
            and (skeleton_type is not None)
            and (model_type is not None)
        ):
            result[dataset][split][skeleton_type][model_type].append(experiment)

    return result


def show_inclusion_table(experiments: List[Experiment], show_empty=True):

    extra_flags = set()
    value_flags = set()

    for experiment in experiments:
        for flag in experiment.flags:

            if isinstance(flag, tuple):
                flag = flag[0]
                value_flags.add(flag)

            if not (
                (flag in DATASET_FLAGS)
                or (flag in SPLIT_FLAGS)
                or (flag in SKELETON_TYPE_FLAGS)
                or (flag in MODEL_TYPE_FLAGS)
            ):
                extra_flags.add(flag)

    extra_flags = list(extra_flags)

    groups = group_experiments(experiments)

    headers = [
        "dataset",
        "split",
        "skeleton",
        "model type",
        "network",
        "top1 acc",
        "top5 acc",
        "samples",
        "batch",
        "test samples",
        *extra_flags,
    ]
    table = []
    raw_table = []

    def colored_line(color, line):

        if color is None:
            return line

        return [color(a) for a in line]

    def experiment_color(experiment: Experiment):

        color = None

        if experiment.age_days <= 0:
            color = colors.magenta
        elif experiment.age_days <= 3:
            color = colors.green
        elif experiment.age_days <= 7:
            color = colors.yellow

        return color

    table.append(colored_line(colors.cyan, headers))
    raw_table.append(headers)
    table.append(SEPARATING_LINE)
    raw_table.append(SEPARATING_LINE)

    last_table_len = 2

    for dataset in DATASET_FLAGS:
        for split in DATASET_SPLIT_FLAGS[dataset]:
            for skeleton_type in SKELETON_TYPE_FLAGS:
                for model_type in MODEL_TYPE_FLAGS:

                    experiments_exist = False

                    experiments = groups[dataset][split][skeleton_type][model_type]
                    experiments = sorted(
                        experiments,
                        key=lambda experiment: -experiment.best_result().top1,
                    )

                    for i, experiment in enumerate(experiments):

                        experiments_exist = True

                        best_result = experiment.best_result()

                        def flag_to_text(f):
                            if f in value_flags:
                                for ef in experiment.flags:
                                    if isinstance(ef, tuple) and ef[0] == f:
                                        return ef[1]
                            else:
                                return "+" if f in experiment.flags else ""

                        line = [
                            dataset,
                            split,
                            skeleton_type,
                            model_type,
                            experiment.network_type,
                            str(best_result.top1) + ("#" if i == 0 else ""),
                            best_result.top5,
                            ("" if experiment.samples is None else experiment.samples),
                            ("" if experiment.batch is None else experiment.batch),
                            ("" if best_result.samples == -1 else best_result.samples),
                            *[flag_to_text(f) for f in extra_flags],
                        ]
                        table.append(colored_line(experiment_color(experiment), line))
                        raw_table.append(line)

                    if (not experiments_exist) and show_empty:

                        line = [
                            dataset,
                            split,
                            skeleton_type,
                            model_type,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            *["" for _ in extra_flags],
                        ]
                        table.append(line)
                        raw_table.append(line)

                table.append(SEPARATING_LINE)
                raw_table.append(SEPARATING_LINE)

            if len(table) - last_table_len > 20:
                table.append(colored_line(colors.cyan, headers))
                raw_table.append(headers)
                table.append(SEPARATING_LINE)
                raw_table.append(SEPARATING_LINE)

                last_table_len = len(table)

    tab = tabulate(table)
    raw_tab = tabulate(raw_table)
    print(tab)
    with open("inclusion_table.txt", "w") as f:
        print(raw_tab, file=f)


def draw_experiments(experiments: List[Experiment]):
    pass


def main(root="./runs", draw=True, show_inclusion=True):
    subdirs = os.walk(root)

    all_result_files = []

    for subdir, _, files in subdirs:
        for file in files:
            if "test.result" in file:
                all_result_files.append((subdir, file))

    print(f"Found {len(all_result_files)} results")

    all_experiments = []

    for subdir, file in all_result_files:
        full_file_name = os.path.join(subdir, file)
        print(full_file_name)

        groups = subdir.replace(root + "/", "").split("/")
        network_type, *params = re.findall(
            r"[a-zA-Z]+|\d+", file.replace("test.result", "")
        )
        samples = None
        batch = None
        flags = []
        iv_type = None

        i = 0

        while i < len(params):
            if params[i] == "s":
                samples = int(params[i + 1])
                i += 1
            elif params[i] == "b":
                batch = int(params[i + 1])
                i += 1
            elif params[i] in ["xufb", "xnfb"]:
                iv_type = params[i] + params[i + 1] + params[i + 2] + params[i + 3]
                flags.append(("iv_type", iv_type))
                i += 3
            elif params[i] in ["xu", "xn"]:
                iv_type = (
                    params[i]
                    + params[i + 1]
                    + params[i + 2]
                    + params[i + 3]
                    + params[i + 4]
                    + params[i + 5]
                )
                flags.append(("iv_type", iv_type))
                i += 5
            elif params[i] in ["f"]:
                iv_type = params[i] + params[i + 1] + params[i + 2] + params[i + 3]
                flags.append(("iv_type", iv_type))
                i += 3
            elif params[i] in ["usual"]:
                iv_type = params[i]
                flags.append(("iv_type", iv_type))
                i += 1
            else:
                flags.append(params[i])
            i += 1

        flags += groups

        experiment_results = []

        with open(full_file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line.replace("'", '"'))
                top1, top5 = re.findall(r"\d+\.\d+", data["result"])

                if "samples" not in data:
                    data["samples"] = -1
                if "batch" not in data:
                    data["batch"] = -1

                single_result = SingleResult(
                    samples=data["samples"],
                    batch=data["batch"],
                    top1=float(top1),
                    top5=float(top5),
                )
                experiment_results.append(single_result)

        experiment = Experiment(
            network_type=network_type,
            samples=samples,
            batch=batch,
            flags=flags,
            results=experiment_results,
            age_days=file_age_in_days(full_file_name),
        )

        all_experiments.append(experiment)

    for exp in all_experiments:
        print(exp)

    if show_inclusion:
        show_inclusion_table(all_experiments)

    if draw:
        draw_experiments(all_experiments)


if __name__ == "__main__":
    Fire(main)
