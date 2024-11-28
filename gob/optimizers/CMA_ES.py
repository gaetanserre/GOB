#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import CMA_ES as C_CMA_ES


class CMA_ES(Optimizer):
    def __init__(self, bounds, n_eval=1000, m_0=None, sigma0=1):
        super().__init__("CMA-ES", bounds)
        self.c_opt = C_CMA_ES(bounds, n_eval, [] if m_0 is None else m_0, sigma0)

    def minimize(self, f):
        return self.c_opt.minimize(f)
