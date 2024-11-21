#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Rosenbrock(Benchmark):
    """
    The Rosenbrock function.
    """

    def __init__(self):
        super().__init__("Rosenbrock", 0)

    def __call__(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
