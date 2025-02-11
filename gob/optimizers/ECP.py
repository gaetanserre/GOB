#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import ECP as C_ECP


class ECP(Optimizer):
    def __init__(
        self, bounds, n_eval=50, epsilon=1e-2, theta_init=1.001, C=1000, verbose=False
    ):
        super().__init__("ECP", bounds)
        self.c_opt = C_ECP(bounds, n_eval, epsilon, theta_init, C, verbose)

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
