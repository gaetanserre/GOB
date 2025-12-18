#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Schwefel(Benchmark):
    """
    The Schwefel function.

    :math:`f(x) = 418.9828d - \\sum_{i = 1}^d x_i \\sin \\left( \\sqrt{|100 x_i + 420.9687|} \\right)`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Schwefel", 0, create_bounds(2, -500, 500))

    def expr(self, x):
        d = len(x)
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
