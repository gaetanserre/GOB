#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import Langevin as C_Langevin


class MSGD(CPP_Optimizer):
    """
    Interface for the Multi-Start Gradient Descent optimizer.

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
        batch_size=0,
        verbose=False,
    ):
        super().__init__("MSGD", bounds, verbose)
        self.c_opt = C_Langevin(bounds, n_particles, iter, dt, 0, batch_size)
