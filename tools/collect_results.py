from math import exp
from unittest import result
from fire import Fire
from pathlib import Path
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
from tabulate import tabulate

DATASET_FLAGS = ["ntu60", "ntu120", "kinetics"]
SPLIT_FLAGS = ["xview", "xsub", ""]
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

    def best_top1(self):
        return max([r.top1 for r in self.results])

    def best_top5(self):
        return max([r.top5 for r in self.results])

    def __str__(self):
        result = f"Experiment(network_type={self.network_type}, samples={self.samples}, batch={self.batch}, flags={self.flags}\n"

        for r in self.results:
            result += f"    {r}\n"

        result += ")"
        return result


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
        
        if (dataset is not None) and (skeleton_type is not None) and (model_type is not None):
            result[dataset][split][skeleton_type][model_type].append(experiment)
    
    return result


def show_inclusion_table(experiments: List[Experiment], show_empty=True):

    extra_flags = set()

    for experiment in experiments:
        for flag in experiment.flags:
            if not ((flag in DATASET_FLAGS) or (flag in SPLIT_FLAGS) or (flag in SKELETON_TYPE_FLAGS) or (flag in MODEL_TYPE_FLAGS)):
                extra_flags.add(flag)
    
    extra_flags = list(extra_flags)

    groups = group_experiments(experiments)

    headers = ["dataset", "split", "skeleton", "model type", "network", "top1 acc", "top5 acc", "samples", "batch", *extra_flags]
    table = []

    for dataset in DATASET_FLAGS:
        for split in SPLIT_FLAGS:
            for skeleton_type in SKELETON_TYPE_FLAGS:
                for model_type in MODEL_TYPE_FLAGS:
                    
                    experiments_exist = False

                    for experiment in groups[dataset][split][skeleton_type][model_type]:

                        experiments_exist = True

                        line = [
                            dataset,
                            split,
                            skeleton_type,
                            model_type,
                            experiment.network_type,
                            experiment.best_top1(),
                            experiment.best_top5(),
                            experiment.samples,
                            experiment.batch,
                            *["+" if f in experiment.flags else "" for f in extra_flags]
                        ]
                        table.append(line)
                    
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
                            *["" for _ in extra_flags]
                        ]
                        table.append(line)

    tab = tabulate(table, headers=headers)
    print(tab)
    with open("inclusion_table.txt", "w") as f:
        print(tab, file=f)

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

        i = 0

        while i < len(params):
            if params[i] == "s":
                samples = int(params[i + 1])
                i += 1
            elif params[i] == "b":
                batch = int(params[i + 1])
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
