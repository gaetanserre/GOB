#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob import create_bounds
from gob.benchmarks import PyGKLS

if __name__ == "__main__":
    opt = {
        "AdaRankOpt": {
            "n_eval": 1000,
            "max_trials": 800,
            "max_degree": 80,
            "verbose": True,
        },
        "SBS": {"n_particles": 200, "svgd_iter": 100},
        "Proportion": {"p": 0.9},
    }
    pygkls = PyGKLS(2, 5, [-5, 5], -20, smoothness="D", deterministic=True)
    gob = GOB(
        ["AdaRankOpt", "AdaLIPO+"],
        ["Levy"],
        ["Proportion"],
        bounds=create_bounds(2, -10, 10, 2),
        options=opt,
    )
    gob.run(n_runs=5, verbose=True)
