#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
import numpy as np
import cma


class CMA_ES(Optimizer):
    def __init__(self, bounds, m_0=None, n_eval=1000, sigma0=1):
        super().__init__("CMA-ES", bounds)

        if m_0 is None:
            d = len(bounds)
            m_0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(d))
        self.m_0 = m_0
        self.n_eval = n_eval
        self.sigma0 = sigma0

    @staticmethod
    def transform_bounds(bounds):
        lo = [0] * len(bounds)
        up = [0] * len(bounds)
        for i, bounds in enumerate(bounds):
            lo[i] = bounds[0]
            up[i] = bounds[1]
        return [lo, up]

    def minimize(self, f):
        res = cma.fmin(
            f,
            self.m_0,
            self.sigma0,
            {
                "bounds": self.transform_bounds(self.bounds),
                "verbose": -9,
                "maxiter": self.n_eval,
            },
        )
        return res[1]
