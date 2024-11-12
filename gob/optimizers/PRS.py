#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .optimizer import Optimizer


class PRS(Optimizer):
    def __init__(self, bounds, n_eval=1000):
        super().__init__("PRS", bounds)
        self.n_eval = n_eval

    def minimize(self, f):
        d = len(self.bounds)
        x = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.n_eval, d)
        )
        f_x = [f(xi) for xi in x]
        return np.min(f_x)
