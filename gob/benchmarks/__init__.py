#
# Created in 2024 by Gaëtan Serré
#

from .square import Square
from .ackley import Ackley
from .levy import Levy
from .michalewicz import Michalewicz
from .rastrigin import Rastrigin
from .rosenbrock import Rosenbrock
from .deb import Deb
from .pygkls import PyGKLS

__all__ = [
    "Square",
    "Ackley",
    "Levy",
    "Michalewicz",
    "Rastrigin",
    "Rosenbrock",
    "Deb",
    "PyGKLS",
]
