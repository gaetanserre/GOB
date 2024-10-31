#
# Created in 2024 by Gaëtan Serré
#

from utils.load_algo import load_algo
from utils.load_function import load_function, free_function
from utils.difficulty import Difficulty
from utils.utils import *

import argparse
import sys
import numpy as np


def cli():
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument(
        "--optimizer",
        "-opt",
        default="CMA",
        help="Choose the optimizer",
    )

    parser.add_argument(
        "--dimension", "-d", type=int, default=2, help="Dimension of the problem"
    )

    parser.add_argument(
        "--bounds",
        "-b",
        type=float,
        default=1,
        help="Bounds of the domain hypercube",
    )

    parser.add_argument(
        "--difficulty",
        "-di",
        type=int,
        default=1,
        help="Difficulty of the generated function. 1: Easy, 2: Medium, 3: Hard 4: Very hard",
    )

    parser.add_argument(
        "--n_eval",
        "-ne",
        type=int,
        default=100_000,
        help="Number of evaluations",
    )

    parser.add_argument(
        "--n_runs",
        "-n",
        type=int,
        default=1,
        help="Number of runs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    if 4 < args.difficulty or args.difficulty < 1:
        print("Error! difficulty must be between 1 and 4.")
        sys.exit(1)

    difficulty = list(Difficulty)[args.difficulty - 1]

    nf = np.random.randint(1, 100)
    f = load_function(nf, args.dimension, difficulty, args.bounds)
    x = np.random.uniform(-args.bounds, args.bounds, (1000000, args.dimension))
    import time

    start = time.time()
    for i in range(1000000):
        _ = f(x[i])
    end = time.time()
    print((end - start) / 1000000)

    """ print_bright_red(
        f"{args.n_runs} runs with {args.optimizer} on a {args.dimension}-dimensional problem with hypercube bounds [{-args.bounds}, {args.bounds}] and difficulty {difficulty.name}."
    )

    mean_approx = 0
    for i in range(args.n_runs):
        nf = np.random.randint(1, 100)
        f = load_function(nf, args.dimension, difficulty, args.bounds)
        bounds = create_bounds(-args.bounds, args.bounds, args.dimension)
        algo = load_algo(args.optimizer, bounds, args.n_eval)

        best, _, _ = algo.optimize(f)
        mean_approx += best[1]

        if i % 2 == 0:
            p_color = print_blue
        else:
            p_color = print_pink
        p_color(f"Run {i + 1} - Best approximation: {best[1]}")

        free_function()
    print_green(f"Mean approximation: {mean_approx / args.n_runs}") """
