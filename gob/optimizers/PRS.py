#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import PRS as C_PRS


class PRS(Optimizer):
    def __init__(self, bounds, n_eval=1000):
        super().__init__("PRS", bounds)
        self.c_opt = C_PRS(bounds, n_eval)

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def get_best_per_iter(self):
        return self.c_opt.get_best_per_iter()
