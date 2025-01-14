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

    def set_stop_criteria(self, stop_criteria):
        self.c_opt.set_stop_criteria(stop_criteria)

    def __del__(self):
        del self.c_opt
