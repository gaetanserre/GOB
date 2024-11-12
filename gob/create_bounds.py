#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def create_bounds(n, mi, ma, d):
    """
    Create bounds for the search space.

    Parameters
    ----------
    n : int
        The number of benchmarks.
    mi : float
        The minimum value of the bounds.
    ma : float
        The maximum value of the bounds.
    d : int
        The number of variables.

    Returns
    -------
    array_like of shape (d, 2)
        The bounds of the search space.
    """
    return np.array([[[mi, ma]] * d] * n)
