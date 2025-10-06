#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Rastrigin(Benchmark):
    """
    The Rastrigin function.

    :math:`f(x) = \\sum_{i=1}^d [x_i ^2 + 10 - 10\\cos(2\\pi x_i)]`

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Rastrigin", 0, create_bounds(2, -5.12, 5.12))

    def expr(self, x):
        return 10 * x.shape[0] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
