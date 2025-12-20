#
# Created in 2025 by Gaëtan Serré
#

from utils import (
    print_avg_rank,
    print_competitive_ratios,
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
sigma = 1 / n_particles**2
n_runs = 20

if __name__ == "__main__":
    algorithms = [
        ("SBS", {"n_particles": n_particles, "iter": iter, "sigma": sigma}),
        (
            "SMD-SBS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
                "moment": "M1",
            },
        ),
        (
            "SMD-SBS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
                "moment": "M2",
            },
        ),
        (
            "SMD-SBS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
                "moment": "VAR",
            },
        ),
        (
            "SMD-SBS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
                "moment": "MVAR",
            },
        ),
        (
            "GCN-SBS",
            {
                "n_particles": n_particles,
                "iter": iter,
                "sigma": sigma,
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
    res_dict, ratios = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="SBS"
    )
    print_avg_rank(res_dict)
    print_competitive_ratios(ratios)

    # flat
    gob = GOB(
        algorithms,
        flat_functions,
        [],
        bounds=flat_functions_bounds,
    )
    print("Running flat functions experiments...")
    res_dict, ratios = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="MSGD"
    )
    print_avg_rank(res_dict)
    print_competitive_ratios(ratios)

    # smooth
    gob = GOB(
        algorithms,
        smooth_functions,
        [],
        bounds=smooth_functions_bounds,
    )
    print("Running smooth functions experiments...")
    res_dict, ratios = gob.run(
        n_runs=n_runs, verbose=1, latex_table=True, reference_optimizer="MSGD"
    )
    print_avg_rank(res_dict)
    print_competitive_ratios(ratios)
