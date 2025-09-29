#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Dixonprice(Benchmark):
    """
    The Dixon-Price function.
    """

    def __init__(self):
        super().__init__("Dixon-Price", 0)

    def expr(self, x):
        return (
            (x[0] - 1) ** 2
            + np.dot(np.arange(2, len(x)+1), (2*x[1:]**2 - x[:-1])**2)
        )