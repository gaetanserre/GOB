#
# Created in 2025 by Gaëtan Serré
#

from ....cpp_optimizer import CPP_Optimizer
from ....cpp_optimizers import GCN_Langevin as CGCN_Langevin


class GCN_Langevin(CPP_Optimizer):
    """
    Interface for the Geometric Common Noise Langevin optimizer.

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
        The inverse temperature.
    sigma_noise : float
        The kernel bandwidth for the common noise.
    independent_noise : bool
        Whether to use independent noise for each particle.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=10,
        beta=1,
        sigma_noise=1,
        independent_noise=True,
        verbose=False,
    ):
        super().__init__("GCN-Langevin", bounds, verbose)

        self.c_opt = CGCN_Langevin(
            bounds, n_particles, iter, dt, beta, sigma_noise, independent_noise
        )
