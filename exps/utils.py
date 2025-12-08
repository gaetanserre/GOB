#
# Created in 2025 by Gaëtan Serré
#

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import gob.benchmarks as gb
from gob.benchmarks import create_bounds, augment_dimensions


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


dim = 50

noisy_functions = [
    gb.Ackley(),
    gb.Deb(),
    gb.Levy(),
    gb.Rastrigin(),
    gb.Rosenbrock(),
    gb.Styblinskitang(),
]

noisy_functions_bounds = [
    augment_dimensions(f.visual_bounds, dim) for f in noisy_functions
]

noisy_functions.append(gb.PyGKLS(dim, 15, [-100, 100], -100, smoothness="ND"))
noisy_functions_bounds.append(create_bounds(dim, -99, 99))

flat_functions = [
    gb.Bentcigar(),
    gb.Dixonprice(),
    gb.Michalewicz(),
    gb.Rosenbrock(),
    gb.Zakharov(),
]
flat_functions_bounds = [
    augment_dimensions(f.visual_bounds, dim) for f in flat_functions
]

smooth_functions = [
    gb.Hyperellipsoid(),
    gb.Square(),
    gb.Sumpow(),
    gb.Trid(),
]
smooth_functions_bounds = [
    augment_dimensions(f.visual_bounds, dim) for f in smooth_functions
]
