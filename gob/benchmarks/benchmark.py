#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


class Benchmark:
    """
    Interface for a benchmark function.

    Parameters
    ----------
    name : str
        The name of the function.
    min : float
        The global minimum of the function.
    """

    def __init__(self, name, min):
        self.name = name
        self.min = min

    def __call__(self, x):
        """
        Evaluate the function at a given point.

        Parameters
        ----------
        x : array_like
            The point at which to evaluate the function.
        """
        pass

    def gradient(self, x):
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
