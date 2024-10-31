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
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (f(x_plus) - f(x)) / eps
    return grad
