#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .metric import Metric


class f_target(Metric):
    """
    Metric that compute the :math:`f`-target metric for a given function and a given proportion.

    Parameters
    ----------
    f : Callable
        The objective function.
    bounds : np.ndarray
        The bounds of the search space.
    p : float
        The proportion.
    """

    def __init__(self, f, bounds, p=0.99):
        super().__init__("f_target")
        self.f = f
        self.bounds = bounds
        self.p = p

    def __call__(self):
        d = self.bounds.shape[0]
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (1_000_000, d))
        n = self.f.n
        fx = [self.f(xi) for xi in x]
        self.f.n = n
        if not hasattr(self.f, "min") or self.f.min is None:
            mn = np.min(fx)
            self.f.min = mn
        else:
            mn = self.f.min
        mean_val = np.mean(fx)
        return mn + (mean_val - mn) * (1 - self.p)
