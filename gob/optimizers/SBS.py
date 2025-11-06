#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import SBS as C_SBS


class SBS(Optimizer):
    """
    Interface for the SBS optimizer.

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
    sigma : float
        The kernel bandwidth.
    alpha : float
        The coefficient to decrease the step size.
    batch_size : int
        The batch size for the mini-batch optimization. If 0, no mini-batch
        optimization is used.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=0.01,
        k=10_000,
        sigma=0.1,
        alpha=0.99,
        batch_size=0,
        verbose=False,
    ):
        super().__init__("SBS", bounds)
        self.c_opt = C_SBS(bounds, n_particles, iter, dt, k, sigma, alpha, batch_size)
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
