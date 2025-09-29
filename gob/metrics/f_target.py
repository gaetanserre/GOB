#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def f_target(f, bounds, p):
    """
    Compute the :math:`f`-target metric for a given function and a given proportion.

    Parameters
    ----------
    f : Callable
        The objective function.
    bounds : np.ndarray
        The bounds of the search space.
    p : float
        The proportion.
    """
    d = bounds.shape[0]
    x = np.random.uniform(bounds[:, 0], bounds[:, 1], (1_000_000, d))
    n = f.n
    fx = [f(xi) for xi in x]
    f.n = n
    if not hasattr(f, "min") or f.min is None:
        mn = np.min(fx)
        f.min = mn
    else:
        mn = f.min
    mean_val = np.mean(fx)
    return mn + (mean_val - mn) * (1 - p)
