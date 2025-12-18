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
        # Compute ranks with handling ties
        ranks = {}
        current_rank = 1
        for i, (optim_name, mean) in enumerate(optim_mean):
            if i > 0 and mean == optim_mean[i - 1][1]:
                ranks[optim_name] = ranks[optim_mean[i - 1][0]]
            else:
                ranks[optim_name] = current_rank
            current_rank += 1
        print(f"Ranks for {bm}:")
        for optim_name, mean in optim_mean:
            rank = ranks[optim_name]
            print(f"  Rank {rank}: {optim_name} with mean {mean:.4f}")
            avg_ranks[optim_name].append(rank)
        print("")

    print("Average ranks over all benchmarks:")
    avg_ranks_list = []
    for optim_name in optims_names:
        avg_rank = np.mean(avg_ranks[optim_name])
        print(f"  {optim_name}: {avg_rank:.2f}")
        avg_ranks_list.append(avg_rank)

    latex_str = "Avg Rank & "
    for avg_rank in avg_ranks_list:
        if avg_rank == np.min(avg_ranks_list):
            latex_str += r"$\mathbf{" + f"{avg_rank:.2f}" + "}$ & "
        else:
            latex_str += f"${avg_rank:.2f}$ & "
    latex_str = (
        latex_str[:-2]
        + " \\\\"
        + "\n"
        + r"\cline{1-"
        + str(len(avg_ranks_list) + 1)
        + "}"
    )
    print("\nLaTeX format:")
    print(latex_str)


dim = 50

noisy_functions = [
    gb.Ackley(),
    gb.Deb(),
    gb.Griewank(),
    gb.Langermann(dim=dim),
    gb.Levy(),
    gb.Rastrigin(),
    gb.Schwefel(),
    gb.Styblinskitang(),
]
noisy_functions_bounds = [
    augment_dimensions(f.visual_bounds, dim) for f in noisy_functions
]

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
