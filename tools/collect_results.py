from fire import Fire
from pathlib import Path
import os
import re
from dataclasses import dataclass
from typing import List
import json


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

    def __str__(self):
        result = f"Experiment(network_type={self.network_type}, samples={self.samples}, batch={self.batch}, flags={self.flags}\n"

        for r in self.results:
            result += f"    {r}\n"
        
        result += ")"
        return result


def main(root="./runs"):
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


if __name__ == "__main__":
    Fire(main)
