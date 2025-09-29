#
# Created in 2025 by Gwendal Debaussart-Joniec
#

import numpy as np
from .benchmark import Benchmark


class Zakharov(Benchmark):
    """
    The Zakharov function.
    """

    def __init__(self):
        super().__init__("Zakharov", 0)

    def expr(self, x):
        return (
            - np.sum(x**2)
            + np.dot(np.arange(1, len(x) + 1), x/2)**2
            + np.dot(np.arange(1, len(x) + 1), x/2)**4
        )
