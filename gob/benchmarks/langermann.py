#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Langermann(Benchmark):
    """
    The Langermann function.

    :math:`f(x) = \\sum_{i = 1}^m c_i \\exp \\left( -\\frac{1}{\\pi} \\sum_{j = 1}^d (x_j - A_{ij})^2 \\right) \\cos \\left( \\pi \\sum_{j = 1}^d (x_j - A_{ij})^2 \\right)`.
    By default, hyper-parameters :math:`m`, :math:`A`, and :math:`c` are generated randomly with a fixed seed.

    Its minimum is unknown.
    """

    def __init__(self, dim=2, m=None, c=None, A=None, seed=42):
        super().__init__("Langermann", 0, create_bounds(2, 0, 10))

        np.random.seed(seed)
        if m is None:
            self.m = np.random.randint(2, 10)
        if c is None:
            self.c = np.random.uniform(-10, 10, self.m)
        if A is None:
            self.A = np.random.uniform(0, 10, (self.m, dim))

    def expr(self, x):
        sum_sq = np.sum((x - self.A) ** 2, axis=1)
        return np.sum(self.c * np.exp(-sum_sq / np.pi) * np.cos(np.pi * sum_sq))
