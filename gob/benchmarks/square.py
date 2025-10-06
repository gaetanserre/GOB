#
# Created in 2024 by Gaëtan Serré
#

from .benchmark import Benchmark
from .create_bounds import create_bounds


class Square(Benchmark):
    """
    The d-square function.

    :math:`f(x) = x \\cdot x^\\top`.

    Its minimum is :math:`0` achieved at :math:`x = 0`.
    """

    def __init__(self):
        super().__init__("Square", 0, create_bounds(2, -10, 10))

    def expr(self, x):
        return x.T @ x

    def gradient(self, x):
        return 2 * x, self(x)
