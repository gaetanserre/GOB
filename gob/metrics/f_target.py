#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def f_target(f, bounds, p):
    """
    Compute the f-target metric for a given function and a given proportion.

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
    x = np.random.uniform(bounds[:, 0], bounds[:, 1], (100_000, d))
    fx = [-f(xi) for xi in x]
    if not hasattr(f, "min") or f.min is None:
        mx = np.max(fx)
        f.min = -mx
    else:
        mx = -f.min
    mean_val = np.mean(fx)
    return -(mx - (mx - mean_val) * (1 - p))
