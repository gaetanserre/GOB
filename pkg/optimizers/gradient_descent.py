#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .optimizer import Optimizer


class GD(Optimizer):
    def __init__(self, bounds, n_step=1000, step_size=1e-3):
        super().__init__("GD", bounds)
        self.n_step = n_step
        self.step_size = step_size

    def minimize(self, f):
        d = len(self.bounds)
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(d))
        for _ in range(self.n_step):
            x -= self.step_size * f.gradient(x)
        return f(x)
