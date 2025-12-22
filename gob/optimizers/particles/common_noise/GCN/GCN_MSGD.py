from ....cpp_optimizer import CPP_Optimizer
from ....cpp_optimizers import GCN_Langevin as CGCN_Langevin


class GCN_MSGD(CPP_Optimizer):
    """
    Interface for the Geometric Common Noise Multi-Start Gradient Descent optimizer.

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
        dt=0.01,
        sigma_noise=1,
        verbose=False,
    ):
        super().__init__("GCN-MSGD", bounds, verbose)

        self.c_opt = CGCN_Langevin(bounds, n_particles, iter, dt, 0, sigma_noise, False)
