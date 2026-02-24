#
# Created in 2025 by Gaëtan Serré
#

from ....cpp_optimizer import CPP_Optimizer
from ....cpp_optimizers import GCN_PSO as CGCN_PSO


class GCN_PSO(CPP_Optimizer):
    """
    Interface for the Geometric Common Noise PSO optimizer.

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
    alpha : float
        The coefficient to decrease the step size.
    sigma_noise : float
        The kernel bandwidth for the common noise.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=1000,
        dt=0.1,
        beta=1e5,
        alpha=1,
        sigma_noise=1,
        verbose=False,
    ):
        super().__init__("GCN-PSO", bounds, verbose)

        self.c_opt = CGCN_PSO(
            bounds,
            n_particles,
            iter,
            dt,
            beta,
            alpha,
            sigma_noise,
        )
