#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import PRS as C_PRS


class PRS(CPP_Optimizer):
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
        super().__init__("PRS", bounds, verbose)
        self.c_opt = C_PRS(bounds, n_eval)
