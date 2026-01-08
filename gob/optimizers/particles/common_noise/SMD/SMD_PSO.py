#
# Created in 2025 by Gaëtan Serré
#

from .SMD_optimizer import SMD_Optimizer
from ....cpp_optimizers import SMD_PSO as CSMD_PSO


class SMD_PSO(SMD_Optimizer):
    """
    Interface for the Stochastic Moment Dynamics PSO optimizer.

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
        iter=1000,
        dt=0.1,
        beta=1e5,
        alpha=1,
        gamma=1,
        lambda_=1e-10,
        delta=2.1,
        moment="M1",
        verbose=False,
    ):
        super().__init__("PSO", bounds, moment, verbose)

        self.c_opt = CSMD_PSO(
            bounds,
            n_particles,
            iter,
            dt,
            beta,
            alpha,
            gamma,
            lambda_,
            delta,
            self.moment,
        )
