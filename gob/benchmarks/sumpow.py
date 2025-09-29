#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Sumpow(Benchmark):
    """
    The Sum of different power function.
    :math:`x : \to \sum_{i=1}^d |x_i|^{i+1}`
    """

    def __init__(self):
        super().__init__("Sumpow", 0)

    def expr(self, x):
        return np.sum(np.abs(x) ** (np.arange(1, len(x) + 1) + 1))