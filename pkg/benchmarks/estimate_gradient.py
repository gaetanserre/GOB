#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


def estimate_gradient(f, x, eps=1e-5):
    """
    Estimate the gradient of a function at a given point using finite differences.

    Parameters
    ----------
    f : callable
        The function to estimate the gradient of. It should take a single argument.
    x : array_like
        The point at which to estimate the gradient of `f`.
    eps : float, optional
        The perturbation used to estimate the gradient.

    Returns
    -------
    array_like
        The estimated gradient of `f` at `x`.
    """
    f_x = f(x)
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps
    return grad
