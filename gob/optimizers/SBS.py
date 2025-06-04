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
        iter=100,
        k=10_000,
        sigma=0.1,
        dt=0.01,
        batch_size=0,
        verbose=False,
    ):
        """
        Interface for the SBS optimizer.

        Parameters
        ----------
        bounds : ndarray
            The bounds of the search space.
        n_particles : int
            The number of particles.
        iter : int
            The number of iterations for the SVGD algorithm.
        k : list
            The kappa exponent.
        sigma : float
            The kernel bandwidth.
        dt : float
            The time step for the SVGD algorithm.
        verbose : bool
            Whether to print information about the optimization process.
        """
        super().__init__("SBS", bounds)
        self.c_opt = C_SBS(bounds, n_particles, iter, k, sigma, dt, batch_size)
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
