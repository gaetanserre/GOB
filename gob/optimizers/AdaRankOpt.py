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
        max_samples=800,
        max_degree=15,
        trust_region_radius=0.1,
        bobyqa_eval=10,
        verbose=False,
    ):
        super().__init__("AdaRankOpt", bounds)

        if n_eval // bobyqa_eval < 2:
            raise ValueError(
                "The number of evaluations should be at least twice the number of evaluations for the BOBYQA optimizer"
            )

        self.c_opt = C_AdaRankOpt(
            bounds,
            n_eval // bobyqa_eval,
            max_samples,
            max_degree,
            trust_region_radius,
            bobyqa_eval,
            verbose,
        )

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criteria(self, stop_criteria):
        self.c_opt.set_stop_criteria(stop_criteria)

    def __del__(self):
        del self.c_opt
