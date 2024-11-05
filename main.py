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
        ["PRS", "GD", "CMA-ES", "AdaLIPO_P", "SBS"],
        ["Square", "Ackley"],
        ["Proportion"],
        bounds=create_bounds(2, -10, 10, 3),
        options=opt,
    )
    gob.run(n_runs=10, verbose=True)
