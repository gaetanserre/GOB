#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Styblinskitang(Benchmark):
    """
    The Styblinski-Tang function.
    This function is normalized by the dimension of x to ensure stability of the minimum.

    :math:`f(x) = \\frac{1}{2d}\\sum_{i=1}^d (x_i^4 - 16 x_i^2 + 5x_i)`.

    Its minimum is :math:`-39.16599` achieved at :math:`x_i = -2.903534`.
    """

    def __init__(self):
        super().__init__("Styblinski-Tang", -39.16599, create_bounds(2, -5, 5))

    def expr(self, x):
        return np.sum(x**4 - 16 * x**2 + 5 * x) / (2 * len(x))
