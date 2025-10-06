#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Rosenbrock(Benchmark):
    """
    The Rosenbrock function.

    :math:`f(x) = \\sum_{i=1}^{d-1} \\left[100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \\right]`

    Its minimum is :math:`0` achieved at :math:`x_i = 1`.
    """

    def __init__(self):
        super().__init__("Rosenbrock", 0, create_bounds(2, -5, 10))

    def expr(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
