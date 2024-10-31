#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .metric import Metric
from .f_target import f_target


class Proportion(Metric):
    """
    Metric that computes the proportion of runs that reached the f-target value.

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

    def __init__(self, f, bounds, p):
        super().__init__("Proportion")
        self.f = f
        self.bounds = bounds
        self.p = p

    def __call__(self, sols):
        target = f_target(self.f, self.bounds, self.p)
        return np.mean([sol <= target for sol in sols])
