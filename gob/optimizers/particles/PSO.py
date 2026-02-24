#
# Created in 2025 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import PSO as C_PSO


class PSO(CPP_Optimizer):
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
        The inverse temperature for using a Gibbs measure to select the global best instead of the argmin. Default is 0 (no Gibbs measure).
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
        iter=1000,
        dt=0.1,
        beta=1e5,
        alpha=1,
        batch_size=0,
        verbose=False,
    ):
        super().__init__("PSO", bounds, verbose)
        self.c_opt = C_PSO(bounds, n_particles, iter, dt, beta, alpha, batch_size)
