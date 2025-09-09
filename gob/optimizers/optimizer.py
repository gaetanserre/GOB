#
# Created in 2024 by Gaëtan Serré
#


class Optimizer:
    """
    Interface for an optimizer.
    """

    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds

    def minimize(self, f):
        """
        Minimize a function using the optimizer.

        Parameters
        ----------
        f : Function
            The objective function.

        Returns
        -------
        pair
            The minimum point and the minimum value.
        """
        pass

    def maximize(self, f):
        """
        Maximize a function using the optimizer.

        Parameters
        ----------
        f : Function
            The objective function.

        Returns
        -------
        pair
            The maximum point and the maximum value.
        """
        f_ = lambda x: -f(x)
        res = self.minimize(f_)
        return res[0], -res[1]

    def verbose_function(self, f):
        """
        Print the value of the function at each evaluation.

        Parameters
        ----------
        f : Function
            The objective function.

        Returns
        -------
        function
            A function that prints the value of the function at each evaluation.
        """
        i = 1

        def f_(*args):
            nonlocal i
            r = f(*args)
            print(f"{self.name} eval #{i} : {r}")
            i += 1
            return r

        return f_

    def set_stop_criterion(self, stop_criterion):
        """
        Set a stop criterion for the optimizer.

        Parameters
        ----------
        stop_criterion : float
            The stop criterion.
        """
        pass

    def __str__(self):
        return self.name
