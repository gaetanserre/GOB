#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Bentcigar(Benchmark):
    """
    The Bent cigar function.

    :math:`f(x) = x_1^2 + 10^6 \\sum_{i=2}^d x_i^2`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Bent cigar", 0)

    def expr(self, x):
        return x[0]**2 + 10 ** 6 * np.sum(x[1:]**2)
