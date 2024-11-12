#
# Created in 2024 by Gaëtan Serré
#


class Metric:
    """
    Interface for a metric.

    Parameters
    ----------
    name : str
        The name of the metric.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, sols):
        """
        Compute the metric.

        Parameters
        ----------
        sols : List
            The solutions returned by a solver during multiple runs.
        """
        pass

    def __str__(self):
        return self.name
