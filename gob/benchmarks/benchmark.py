#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from typing import Tuple


class Benchmark:
    """
    Interface for a benchmark function.

    Parameters
    ----------
    name : str
        The name of the function.
    min : float
        The global minimum of the function.
    visual_bounds : list of list of float, optional
        The bounds for visualization in 2D (default is None).
    """

    def __init__(self, name, min, visual_bounds=None):
        self.name = name
        self.min = min
        self.n = 0
        self.visual_bounds = visual_bounds

    def expr(self, x):
        """
        The expression of the function.

        Parameters
        ----------
        x : array-like
            The point at which to evaluate the function.

        Returns
        -------
        float
            The value of the function at `x`.
        """
        pass

    def __call__(self, x):
        """
        Evaluate the function at a given point.

        Parameters
        ----------
        x : array-like
            The point at which to evaluate the function.

        Returns
        -------
        float
            The value of the function at `x`.
        """
        self.n += 1
        return self.expr(x)

    def gradient(self, x):
        """
        Estimate the gradient of a function at a given point using finite differences.

        Parameters
        ----------
        f : callable
            The function to estimate the gradient of. It should take a single argument.
        x : array-like
            The point at which to estimate the gradient of `f`.
        eps : float, optional
            The perturbation used to estimate the gradient.

        Returns
        -------
        pair
            The estimated gradient of `f` at `x` and the value of `f` at `x`.
        """
        eps = 1e-12
        f_x = self(x)
        grad = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_p = x.copy()
            x_p[i] += eps
            grad[i] = (self(x_p) - f_x) / eps
        return grad, f_x

    def __str__(self):
        return self.name
