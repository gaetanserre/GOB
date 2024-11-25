#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob import create_bounds
from gob.benchmarks import PyGKLS

if __name__ == "__main__":
    opt = {
        "AdaRankOpt": {"n_eval": 1000},
        "SBS": {"n_particles": 200, "svgd_iter": 100},
        "Proportion": {"p": 0.995},
    }
    pygkls = PyGKLS(2, 10, [-5, 5], -20, smoothness="D", deterministic=True)
    gob = GOB(
        ["AdaRankOpt", "AdaLIPO+", "SBS", "CMA-ES"],
        [pygkls],
        ["Proportion"],
        bounds=create_bounds(2, -4, 4, 2),
        options=opt,
    )
    gob.run(n_runs=1, verbose=True)
