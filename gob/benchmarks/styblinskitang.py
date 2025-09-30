#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Styblinskitang(Benchmark):
    """
    The Styblinski-Tang function.
    This function is normalized by the dimension of x to ensure stability of the minimum.
    :math:`x \\in \\mathbb{R}^d \\to \\frac{1}{2d}\\sum_{i=1}^d (x_i^4 - 16 x_i^2 + 5x_i)`
    """

    def __init__(self):
        super().__init__("Styblinski-Tang", -39.16599)

    def expr(self, x):
        return np.sum(x**4 - 16 * x**2 + 5 * x ) / (2 * len(x))