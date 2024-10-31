#
# Created in 2024 by Gaëtan Serré
#

import pygkls
from .difficulty import Difficulty


def load_function(nf, dim, difficulty, bounds):
    pygkls.init(
        nf=nf, dim=dim, num_minima=difficulty.value, domain_lo=-bounds, domain_hi=bounds
    )

    def f_aux(x):
        if difficulty <= Difficulty.Medium:
            return pygkls.get_d2_f(x)
        elif difficulty == Difficulty.Hard:
            return pygkls.get_d_f(x)
        else:
            return pygkls.get_nd_f(x)

    return f_aux


def free_function():
    pygkls.free()
