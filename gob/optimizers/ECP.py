#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import ECP as C_ECP


class ECP(Optimizer):
    def __init__(
        self,
        bounds,
        n_eval=50,
        epsilon=1e-2,
        theta_init=1.001,
        C=1000,
        max_trials=10_000_000,
        trust_region_radius=0.1,
        bobyqa_eval=10,
        verbose=False,
    ):
        super().__init__("ECP+TR", bounds)

        if n_eval < bobyqa_eval:
            bobyqa_eval = n_eval
            n_eval = 1
        else:
            n_eval = n_eval // bobyqa_eval

        self.c_opt = C_ECP(
            bounds,
            n_eval,
            epsilon,
            theta_init,
            C,
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
