#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import SBS as C_SBS


class SBS(Optimizer):
    def __init__(
        self,
        bounds,
        n_particles=200,
        svgd_iter=100,
        k_iter=[10_000],
        sigma=0.1,
        lr=0.5,
    ):
        super().__init__("SBS", bounds)
        self.c_opt = C_SBS(bounds, n_particles, svgd_iter, k_iter, sigma, lr)

    def minimize(self, f):
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
