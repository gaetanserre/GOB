#
# Created in 2024 by Gaëtan Serré
#


class Optimizer:
    """
    Interface for an optimizer.
    """

    def __init__(self, name, bounds):
        """
        Initialize the optimizer.

        Parameters
        ----------
        name : str
            The name of the optimizer.
        bounds : array_like of shape (n_variables, 2)
            The bounds of the search space.
        """
        self.name = name
        self.bounds = bounds

    def minimize(self, f):
        """
        Minimize a function using the optimizer.

        Parameters
        ----------
        f : Function
            The objective function.
        """
        pass

    def __str__(self):
        return self.name
