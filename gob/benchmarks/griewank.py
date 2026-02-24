#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Griewank(Benchmark):
    """
    The Griewank function.

    :math:`f(x) = \\sum_{i = 1}^d \\frac{x_i^2}{4000} - \\prod_{i = 1}^d \\cos \\left( \\frac{x_i^2}{\\sqrt{i}} \\right) + 1`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Griewank", 0, create_bounds(2, -150, 150))

    def expr(self, x):
        return (
            np.sum(x**2) / 4000
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
            + 1
        )
