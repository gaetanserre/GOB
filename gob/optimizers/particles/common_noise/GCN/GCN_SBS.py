#
# Created in 2025 by Gaëtan Serré
#

from ....cpp_optimizer import CPP_Optimizer
from ....cpp_optimizers import GCN_SBS as CGCN_SBS


class GCN_SBS(CPP_Optimizer):
    """
    Interface for the Geometric Common Noise SBS optimizer.

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
    sigma : float
        The kernel bandwidth.
    sigma_noise : float
        The kernel bandwidth for the common noise.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=10,
        sigma=0.1,
        sigma_noise=1,
        verbose=False,
    ):
        super().__init__("GCN-SBS", bounds, verbose)

        self.c_opt = CGCN_SBS(
            bounds,
            n_particles,
            iter,
            dt,
            sigma,
            sigma_noise,
        )
