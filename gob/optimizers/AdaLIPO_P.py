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
        max_trials=800,
        trust_region_radius=0.1,
        bobyqa_eval=10,
        verbose=False,
    ):
        super().__init__("AdaLIPO+", bounds)
        if n_eval // bobyqa_eval < 2:
            raise ValueError(
                "The number of evaluations should be at least twice the number of evaluations for the BOBYQA optimizer"
            )

        self.c_opt = C_AdaLIPO_P(
            bounds,
            n_eval // bobyqa_eval,
            max_trials,
            trust_region_radius,
            bobyqa_eval,
            verbose,
        )

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
