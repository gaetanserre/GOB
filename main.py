#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from pkg import GOB
from pkg import create_bounds

if __name__ == "__main__":
    opt = {"PRS": {"n_eval": 1000000}}
    gob = GOB(
        "PRS", "Square", "Proportion", bounds=create_bounds(-10, 10, 3), options=opt
    )
    gob.run(n_runs=10, verbose=True)
