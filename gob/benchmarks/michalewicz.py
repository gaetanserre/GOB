#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Michalewicz(Benchmark):
    """
    The Michalewicz function.
    """

    def __init__(self):
        super().__init__("Michalewicz", None)

    def expr(self, x):
        dim = x.shape[0]
        id_ = np.arange(1, dim + 1)
        return -np.sum(np.sin(x) * np.sin(id_ * x**2 / np.pi) ** (2 * 10))
