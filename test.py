#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob import create_bounds

if __name__ == "__main__":
    opt = {
        "AdaRankOpt": {"n_eval": 1000},
        "SBS": {"n_particles": 200, "svgd_iter": 100},
        "Proportion": {"p": 0.995},
    }
    gob = GOB(
        [
            "AdaLIPO+",
            "AdaRankOpt",
            "BayesOpt",
            "CMA-ES",
            "CRS",
            "Direct",
            "GD",
            "MLSL",
            "PRS",
            "SBS",
        ],
        ["Square"],
        ["Proportion"],
        bounds=create_bounds(2, -1, 1, 2),
        options=opt,
    )
    gob.run(n_runs=1, verbose=True)
