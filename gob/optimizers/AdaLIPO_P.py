#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import AdaLIPO_P as C_AdaLIPO_P


class AdaLIPO_P(Optimizer):
    def __init__(self, bounds, n_eval=1000, window_slope=5, max_slope=600):
        super().__init__("AdaLIPO+", bounds)
        self.c_opt = C_AdaLIPO_P(bounds, n_eval, window_slope, max_slope)

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def get_best_per_iter(self):
        return self.c_opt.get_best_per_iter()
