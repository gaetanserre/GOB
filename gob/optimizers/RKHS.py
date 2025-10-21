#
# Created in 2025 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import RKHS as C_RKHS


class RKHS(Optimizer):
    """
    Interface for the *social only* PSO optimizer.

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
    beta : float
        The inverse temperature for using a Gibbs measure to select the consensus point.
    sigma : float
        The bandwidth of the RBF kernel.
    epsilon : float
        The noise coefficient.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=1000,
        dt=0.01,
        beta=1e5,
        sigma=1,
        epsilon=0.5,
        alpha=1,
        batch_size=0,
        verbose=False,
    ):
        super().__init__("RKHS", bounds)
        self.c_opt = C_RKHS(
            bounds, n_particles, iter, dt, beta, sigma, epsilon, alpha, batch_size
        )
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
