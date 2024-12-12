#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import AdaRankOpt as C_AdaRankOpt


class AdaRankOpt(Optimizer):
    def __init__(
        self,
        bounds,
        n_eval=1000,
        max_degree=8,
        max_samples=800,
        bobyqa=True,
        bobyqa_maxfun=50,
        verbose=False,
    ):
        super().__init__("AdaRankOpt", bounds)
        self.c_opt = C_AdaRankOpt(
            bounds, n_eval, max_degree, max_samples, bobyqa, bobyqa_maxfun, verbose
        )

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criteria(self, stop_criteria):
        self.c_opt.set_stop_criteria(stop_criteria)

    def __del__(self):
        del self.c_opt
