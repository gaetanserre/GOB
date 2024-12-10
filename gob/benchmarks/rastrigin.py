#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Rastrigin(Benchmark):
    """
    The Rastrigin function.
    """

    def __init__(self):
        super().__init__("Rastrigin", 0)

    def expr(self, x):
        return 10 * x.shape[0] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
