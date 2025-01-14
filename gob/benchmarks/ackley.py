#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Ackley(Benchmark):
    """
    The Ackley function.
    """

    def __init__(self):
        super().__init__("Ackley", 0)

    def expr(self, x):
        a = 20
        b = 0.2
        c = 1
        return (
            -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(c * x)) / len(x))
            + a
            + np.e
        )
