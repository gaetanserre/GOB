#
# Created in 2025 by Gaëtan Serré
#

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gob import GOB
from gob.benchmarks import PyGKLS, create_bounds, augment_dimensions
import inspect
import gob.benchmarks as gb
import numpy as np


def print_avg_rank(res_dict):
    bms = list(res_dict.keys())
    optims_names = list(res_dict[bms[0]].keys())
    avg_ranks = {name: [] for name in optims_names}
    for bm in bms:
        optim_mean = []
        for optim_name in res_dict[bm]:
            optim_mean.append((optim_name, res_dict[bm][optim_name]["Approx"]["mean"]))
        optim_mean = sorted(optim_mean, key=lambda x: x[1])
        print(f"Ranks for {bm}:")
        for rank, (optim_name, mean) in enumerate(optim_mean):
            print(f"  Rank {rank + 1}: {optim_name} with mean {mean:.4f}")
            avg_ranks[optim_name].append(rank + 1)
        print("")
    print("Average ranks over all benchmarks:")
    latex_str = "Avg Ranks & "
    for optim_name in optims_names:
        avg_rank = np.mean(avg_ranks[optim_name])
        print(f"  {optim_name}: {avg_rank:.2f}")
        latex_str += f"{avg_rank:.2f} & "
    latex_str = latex_str[:-2] + " \\\\"
    print("\nLaTeX format:")
    print(latex_str)


n_particles = 150
iter = 300
sigma = 1 / n_particles**2
dim = 50

if __name__ == "__main__":
    pygkls = PyGKLS(dim, 15, [-100, 100], -100, smoothness="ND")

    benchmarks = []
    bounds = []
    for name, obj in inspect.getmembers(gb, inspect.isclass):
        if name != "PyGKLS":
            benchmarks.append(obj())
            bounds.append(augment_dimensions(benchmarks[-1].visual_bounds, dim))
    benchmarks.append(pygkls)
    bounds.append(create_bounds(dim, -99, 99))

    gob = GOB(
        [
            ("SBS", {"n_particles": n_particles, "iter": iter}),
            (
                "CN-SBS",
                {
                    "n_particles": n_particles,
                    "iter": iter,
                    "moment": "M1",
                },
            ),
            (
                "CN-SBS",
                {
                    "n_particles": n_particles,
                    "iter": iter,
                    "moment": "M2",
                },
            ),
            (
                "CN-SBS",
                {
                    "n_particles": n_particles,
                    "iter": iter,
                    "moment": "VAR",
                },
            ),
            ("SBS-RKHS", {"n_particles": n_particles, "iter": iter}),
        ],
        benchmarks,
        ["Proportion"],
        bounds=bounds,
    )
    res_dict = gob.run(n_runs=20, verbose=1, latex_table=True)
    print_avg_rank(res_dict)
    print(res_dict)
