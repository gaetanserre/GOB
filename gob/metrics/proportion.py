#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .metric import Metric
from .f_target import f_target


class Proportion(Metric):
    """
    Metric that computes the proportion of runs that reached the :math:`f`-target value.

    Parameters
    ----------
    sols : List
        The solutions returned by a solver during multiple runs.
    f : Callable
        The objective function.
    bounds : np.ndarray
        The bounds of the search space.
    p : float
        The proportion.
    """

    def __init__(self, f, bounds, p=0.99, f_target=None):
        super().__init__("Proportion")
        self.f = f
        self.bounds = bounds
        self.p = p
        self.f_target = f_target

    def __call__(self, sols):
        if self.f_target is None:
            self.f_target = f_target(self.f, self.bounds, self.p)()
        return np.mean([sol <= self.f_target for sol in sols])
