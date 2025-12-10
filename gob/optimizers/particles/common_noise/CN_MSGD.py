#
# Created in 2024 by Gaëtan Serré
#

from .common_noise_optimizer import CN_Optimizer
from ...cpp_optimizers import CN_Langevin as CCN_Langevin


class CN_MSGD(CN_Optimizer):
    """
    Interface for the Common noise Multi-Start Gradient Descent optimizer.

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
    gamma : float
        The coefficient for the common noise.
    ``lambda_`` : float
        The regularization parameter for the common noise.
    delta : float
        The parameter for the Bessel process.
    moment : str
        The type of moment used for the common noise ("M1" | "M2" | "VAR" | "MVAR").
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=10,
        gamma=1,
        lambda_=0,
        delta=2.1,
        moment="M2",
        verbose=False,
    ):
        super().__init__("CN-MSGD", bounds, moment, verbose)

        self.c_opt = CCN_Langevin(
            bounds, n_particles, iter, dt, 0, gamma, lambda_, delta, self.moment
        )
