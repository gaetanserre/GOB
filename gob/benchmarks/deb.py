#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .benchmark import Benchmark


class Deb(Benchmark):
    """
    The Deb N.1 function.
    """

    def __init__(self):
        super().__init__("Deb N.1", -1)

    def expr(self, x):
        return -np.sum(np.sin(5 * np.pi * x) ** 6) / x.shape[0]
