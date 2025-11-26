#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import CMA_ES as C_CMA_ES


class CMA_ES(CPP_Optimizer):
    """
    Interface for the CMA-ES optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_eval : int
        The maximum number of function evaluations.
    m_0 : ndarray
        The initial mean of the search distribution.
    sigma0 : float
        The initial standard deviation of the search distribution.
    verbose : bool
        Whether to print information about the optimization

    """

    def __init__(self, bounds, n_eval=1000, m_0=None, sigma0=1, verbose=False):
        super().__init__("CMA-ES", bounds, verbose)
        self.c_opt = C_CMA_ES(bounds, n_eval, [] if m_0 is None else m_0, sigma0)
