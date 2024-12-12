#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import AdaLIPO_P as C_AdaLIPO_P


class AdaLIPO_P(Optimizer):
    def __init__(
        self,
        bounds,
        n_eval=1000,
        window_slope=5,
        max_slope=600,
        bobyqa=True,
        bobyqa_maxfun=50,
    ):
        super().__init__("AdaLIPO+", bounds)
        self.c_opt = C_AdaLIPO_P(
            bounds,
            n_eval if not bobyqa else (2 * n_eval) // (bobyqa_maxfun + 1),
            window_slope,
            max_slope,
            bobyqa,
            bobyqa_maxfun,
        )

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criteria(self, stop_criteria):
        self.c_opt.set_stop_criteria(stop_criteria)

    def __del__(self):
        del self.c_opt
