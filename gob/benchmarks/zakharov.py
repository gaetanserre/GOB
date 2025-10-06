#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Zakharov(Benchmark):
    """
    The Zakharov function.

    :math:`f(x) = \\sum_{i=1}^d x_i^2 + \\left(\\sum_{i=1}^d \\frac{1}{2} i x_i\\right)^2 + \\left(\\sum_{i=1}^d \\frac{1}{2} i x_i\\right)^4`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Zakharov", 0, create_bounds(2, -5, 10))

    def expr(self, x):
        return (
            -np.sum(x**2)
            + np.dot(np.arange(1, len(x) + 1), x / 2) ** 2
            + np.dot(np.arange(1, len(x) + 1), x / 2) ** 4
        )
