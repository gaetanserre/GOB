#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Sumpow(Benchmark):
    """
    The Sum of different power function.

    :math:`f(x) = \\sum_{i=1}^d |x_i|^{i+1}`

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Sumpow", 0, create_bounds(2, 0, 2))

    def expr(self, x):
        return np.sum(np.abs(x) ** (np.arange(1, len(x) + 1) + 1))
