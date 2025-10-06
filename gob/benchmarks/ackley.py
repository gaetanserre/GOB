#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Ackley(Benchmark):
    """
    The Ackley function.

    :math:`f(x) = -a\\exp\\left(-b \\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2} \\right) - \\exp\\left( \\frac{1}{d} \\sum_{i=1}^d \\cos(c x_i) \\right) + a + \\exp(1)`.
    By default, hyper-parameters are set to :math:`a = 20, b = 0.2, c = 1`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self, a=20, b=0.2, c=1):
        super().__init__("Ackley", 0, create_bounds(2, -32.768, 32.768))
        self.a = a
        self.b = b
        self.c = c

    def expr(self, x):
        return (
            -self.a * np.exp(-self.b * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(self.c * x)) / len(x))
            + self.a
            + np.e
        )
