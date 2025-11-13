#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def create_bounds(d, mi, ma, n=1):
    """
    Create bounds for the search space.

    Parameters
    ----------
    d : int
        The number of variables.
    mi : float
        The minimum value of the bounds.
    ma : float
        The maximum value of the bounds.
    n : int
        The number of benchmarks.
    Returns
    -------
    ndarray of shape (d, 2)
        The bounds of the search space.
    """
    if n == 1:
        return np.array([[mi, ma]] * d)
    else:
        return np.array([[[mi, ma]] * d] * n)


def augment_dimensions(bounds, target_dim):
    """
    Augment the dimensions of the bounds to the target dimension.

    Parameters
    ----------
    bounds : ndarray
        The original bounds of the search space.
    target_dim : int
        The target dimension.
    Returns
    -------
    ndarray of shape (target_dim, 2)
        The augmented bounds of the search space.
    """
    bounds_dim0 = bounds[0]
    return np.array([bounds_dim0] * target_dim)
