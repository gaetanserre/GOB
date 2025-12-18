#
# Created in 2025 by Gaëtan Serré
#

from ....cpp_optimizer import CPP_Optimizer
from ....cpp_optimizers import GCN_CBO as CGCN_CBO


class GCN_CBO(CPP_Optimizer):
    """
    Interface for the Geometric Common Noise CBO optimizer.

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
    lam : float
        The attraction parameter.
    epsilon : float
        The smooth-heaviside parameter.
    beta : float
        The inverse temperature.
    sigma : float
        The standard deviation of the Gaussian noise.
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
        dt=0.01,
        lam=1,
        epsilon=1e-2,
        beta=1,
        sigma=5.1,
        alpha=1,
        sigma_noise=1,
        verbose=False,
    ):
        super().__init__("GCN-CBO", bounds, verbose)

        self.c_opt = CGCN_CBO(
            bounds,
            n_particles,
            iter,
            dt,
            lam,
            epsilon,
            beta,
            sigma,
            alpha,
            sigma_noise,
        )
