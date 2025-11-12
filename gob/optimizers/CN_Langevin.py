#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import CN_Langevin as CCN_Langevin


class CN_Langevin(Optimizer):
    """
    Interface for the Common noise Langevin optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_particles : int
        The number of particles.
    iter : int
        The number of iterations.
    dt : float
        The time step.
    k : list
        The kappa exponent.
    beta : float
        The inverse temperature.
    alpha : float
        The coefficient to decrease the step size.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=0.1,
        k=10_000,
        beta=0.5,
        alpha=1,
        verbose=False,
    ):
        super().__init__("CN_Langevin", bounds)
        self.c_opt = CCN_Langevin(bounds, n_particles, iter, dt, k, beta, alpha)
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
