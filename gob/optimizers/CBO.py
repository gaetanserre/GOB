#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import CBO as C_CBO


class CBO(Optimizer):
    def __init__(
        self,
        bounds,
        n_particles=20,
        iter=1000,
        dt=0.01,
        lam=1,
        epsilon=1e-2,
        beta=1,
        sigma=5.1,
        use_batch=False,
        verbose=False,
    ):
        """
        Interface for the CBO optimizer.

        Parameters
        ----------
        bounds : ndarray
            The bounds of the search space.
        n_particles : int
            The number of particles.
        iter : int
            The number of iterations for the SVGD algorithm.
        """
        super().__init__("CBO", bounds)
        self.c_opt = C_CBO(
            bounds, n_particles, iter, dt, lam, epsilon, beta, sigma, use_batch
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
