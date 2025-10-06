#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Deb(Benchmark):
    """
    The Deb N.1 function.

    :math:`f(x) = -\\frac{1}{d}\\sum_{i=1}^d \\sin(5\\pi x_i)^6`

    Its minimum is :math:`-1`.
    """

    def __init__(self):
        super().__init__("Deb N.1", -1, create_bounds(2, -0.5, 0.5))

    def expr(self, x):
        return -np.sum(np.sin(5 * np.pi * x) ** 6) / x.shape[0]
