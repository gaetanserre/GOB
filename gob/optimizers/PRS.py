#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import PRS as C_PRS


class PRS(Optimizer):
    """
    Interface for the PRS optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_eval : int
        The maximum number of function evaluations.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(self, bounds, n_eval=1000, verbose=False):
        super().__init__("PRS", bounds)
        self.c_opt = C_PRS(bounds, n_eval)
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
