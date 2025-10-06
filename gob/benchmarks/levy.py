#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Levy(Benchmark):
    """
    The Levy function.

    :math:`f(x) = \\sin(\\pi w_1)^2 + \\sum_{i=1}^{d-1}(w_i - 1)^2[1 + 10 \\sin(\\pi w_i +1)^2] + (w_d - 1)^2[1 + sin^2(2\\pi w_d)]`, where :math:`w_i = \\frac{x_i + 3}{4}`.

    Its minimum is :math:`0` achieved at :math:`x_i = 1`.
    """

    def __init__(self):
        super().__init__("Levy", 0, create_bounds(2, -10, 10))

    def expr(self, x):
        w = 1 + (x - 1) / 4
        return (
            np.sin(np.pi * w[0]) ** 2
            + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        )
