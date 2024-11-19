#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob import create_bounds

if __name__ == "__main__":
    opt = {
        "SBS": {"n_particles": 200, "svgd_iter": 100},
        "GD": {"n_step": 3000, "step_size": 1e-3},
        "AdaRankOpt": {"method": "simplex", "n_eval": 100},
        "Proportion": {"p": 0.995},
    }
    gob = GOB(
        ["SBS"],
        ["Square", "Ackley"],
        ["Proportion"],
        bounds=create_bounds(2, -1, 1, 2),
        options=opt,
    )
    gob.run(n_runs=1, verbose=True)
