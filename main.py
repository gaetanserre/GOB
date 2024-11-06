#
# Created in 2024 by Gaëtan Serré
#

from pkg import GOB
from pkg import create_bounds

if __name__ == "__main__":
    opt = {
        "PRS": {"n_eval": 1000},
        "GD": {"n_step": 3000, "step_size": 1e-3},
        "Proportion": {"p": 0.995},
    }
    gob = GOB(
        ["AdaRankOpt"],
        ["Square", "Ackley"],
        ["Proportion"],
        bounds=create_bounds(2, -3, 3, 2),
        options=opt,
    )
    gob.run(n_runs=1, verbose=True)
