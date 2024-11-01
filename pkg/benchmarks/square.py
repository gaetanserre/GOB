#
# Created in 2024 by Gaëtan Serré
#

from .benchmark import Benchmark


class Square(Benchmark):
    """
    The d-square function.
    """

    def __init__(self):
        super().__init__("Square", 0)

    def __call__(self, x):
        return x.T @ x

    def gradient(self, x):
        return 2 * x
