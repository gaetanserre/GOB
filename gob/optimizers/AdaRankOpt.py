#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import AdaRankOpt as C_AdaRankOpt


class AdaRankOpt(Optimizer):
    def __init__(self, bounds, n_eval=1000, simplex_tol=1e-6):
        super().__init__("AdaRankOpt", bounds)
        self.c_opt = C_AdaRankOpt(bounds, n_eval, simplex_tol)

    def minimize(self, f):
        return self.c_opt.minimize(f)
