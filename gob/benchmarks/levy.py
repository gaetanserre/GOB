#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Levy(Benchmark):
    """
    The Levy function.
    """

    def __init__(self):
        super().__init__("Levy", 0)

    def __call__(self, x):
        w = 1 + (x - 1) / 4
        return (
            np.sin(np.pi * w[0]) ** 2
            + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        )
