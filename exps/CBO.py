#
# Created in 2025 by Gaëtan Serré
#

from utils import (
    print_avg_rank,
    noisy_functions,
    noisy_functions_bounds,
    flat_functions,
    flat_functions_bounds,
    smooth_functions,
    smooth_functions_bounds,
)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gob import GOB

n_particles = 150
iter = 300
n_runs = 20

if __name__ == "__main__":
    algorithms = [
        ("CBO", {"n_particles": n_particles, "iter": iter}),
        (
            "SMD-CBO",
            {
                "n_particles": n_particles,
                "iter": iter,
                "moment": "M1",
            },
        ),
        (
            "SMD-CBO",
            {
                "n_particles": n_particles,
                "iter": iter,
                "moment": "M2",
            },
        ),
        (
            "SMD-CBO",
            {
                "n_particles": n_particles,
                "iter": iter,
                "moment": "VAR",
            },
        ),
        (
            "SMD-CBO",
            {
                "n_particles": n_particles,
                "iter": iter,
                "moment": "MVAR",
            },
        ),
        (
            "GCN-CBO",
            {
                "n_particles": n_particles,
                "iter": iter,
            },
        ),
    ]

    # noisy
    gob = GOB(
        algorithms,
        noisy_functions,
        [],
        bounds=noisy_functions_bounds,
    )
    print("Running noisy functions experiments...")
    res_dict = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="CBO"
    )
    print_avg_rank(res_dict)

    # flat
    gob = GOB(
        algorithms,
        flat_functions,
        [],
        bounds=flat_functions_bounds,
    )
    print("Running flat functions experiments...")
    res_dict = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="CBO"
    )
    print_avg_rank(res_dict)

    # smooth
    gob = GOB(
        algorithms,
        smooth_functions,
        [],
        bounds=smooth_functions_bounds,
    )
    print("Running smooth functions experiments...")
    res_dict = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="CBO"
    )
    print_avg_rank(res_dict)
