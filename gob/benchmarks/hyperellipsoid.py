#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Hyperellipsoid(Benchmark):
    """
    The Rotated Hyper-Ellipsoid function.

    :math:`f(x) = \\sum_{i=1}^d \\sum_{j=1}^i x_j^2`

    Its minimum is :math:`0` achieved at :math:`x_i = 0`.
    """

    def __init__(self):
        super().__init__("Hyper-Ellipsoid", 0, create_bounds(2, -5, 5))

    def expr(self, x):
        return np.sum(np.cumsum(x**2))
