#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Hyperellipsoid(Benchmark):
    """
    The Rotated Hyper-Ellipsoid function.
    :math:`x : \\to \\sum_{i=1}^d \\sum_{j=1}^i x_j^2`
    """

    def __init__(self):
        super().__init__("Hyper-Ellipsoid", 0)

    def expr(self, x):
        return np.sum(np.cumsum(x ** 2))