#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark
from .create_bounds import create_bounds


class Michalewicz(Benchmark):
    """
    The Michalewicz function.

    :math:`f(x) =  - \\sum_{i=1}^d \\sin(x_i) \\sin\\left(\\frac{x_i^2}{\\pi}\\right)^{2\\times 10}`

    Its minimum is unknown.
    """

    def __init__(self):
        super().__init__("Michalewicz", None, create_bounds(2, 0, np.pi))

    def expr(self, x):
        dim = x.shape[0]
        id_ = np.arange(1, dim + 1)
        return -np.sum(np.sin(x) * np.sin(id_ * x**2 / np.pi) ** (2 * 10))
