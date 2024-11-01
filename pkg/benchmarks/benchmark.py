#
# Created in 2024 by Gaëtan Serré
#

from .estimate_gradient import estimate_gradient


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
        Evaluate the gradient of the function at a given point.

        Parameters
        ----------
        x : array_like
            The point at which to evaluate the gradient of the function.
        """
        return estimate_gradient(self, x)

    def __str__(self):
        return self.name
