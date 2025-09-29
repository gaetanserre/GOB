#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Sumcumsum(Benchmark):
    """
    The Sum Cum-Sum function, alias Rotated Hyper-Ellipsoid.
    :math:`x : \to \sum_{i=1}^d \sum_{j=1}^i x_j^2`
    """

    def __init__(self):
        super().__init__("Sum Cum-Sum", 0)

    def expr(self, x):
        return np.sum(np.cumsum(x ** 2))