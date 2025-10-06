#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Trid(Benchmark):
    """
    The Trid function.
    The function is normalized according to the dimension to ensure consistency in the global minima.

    :math:`f(x) = \\frac{1}{d(d+4)(d-1)} \\left(\\sum_{i=1}^d (x_i - 1)^2 - \\sum_{i=2}^d x_i x_{i-1}\\right)`.

    Its minimum is :math:`-\\frac{1}{6}` achieved at :math:`x_i = i(d + 1 - i)`.
    """

    def __init__(self):
        super().__init__("Trid", -1 / 6, create_bounds(2, -4, 4))

    def expr(self, x):
        return (np.sum((x - 1) ** 2) - np.sum(x[1:] * x[:-1])) / (
            len(x) * (len(x) + 4) * (len(x) - 1)
        )
