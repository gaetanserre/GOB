#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Dixonprice(Benchmark):
    """
    The Dixon-Price function.

    :math:`f(x) = (x_1 - 1)^2 + \\sum_{i=1}^d i (2x_i^2 - x_{i-1})^2`.

    Its minimum is :math:`0` achieved at :math:`x_i = 2^\\frac{2^i - 2}{2^i}`.
    """

    def __init__(self):
        super().__init__("Dixon-Price", 0, create_bounds(2, -10, 10))

    def expr(self, x):
        return (x[0] - 1) ** 2 + np.dot(
            np.arange(2, len(x) + 1), (2 * x[1:] ** 2 - x[:-1]) ** 2
        )
