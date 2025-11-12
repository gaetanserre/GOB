#
# Created in 2024 by Gaëtan Serré
#

from ..optimizer import Optimizer
from ..cpp_optimizers import CN_Langevin as CCN_Langevin


class CN_Langevin(Optimizer):
    """
    Interface for the Common noise Langevin optimizer.

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
        The type of moment used for the common noise ("M1" | "M2" | "VAR").
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
        dt=0.1,
        beta=0.5,
        alpha=1,
        gamma=1,
        lambda_=0,
        delta=2.1,
        moment="M2",
        verbose=False,
    ):
        super().__init__("CN_Langevin", bounds)

        match moment:
            case "M1":
                moment = 0
            case "M2":
                moment = 1
            case "VAR":
                moment = 2
            case _:
                raise ValueError(
                    'Invalid moment type. Choose from "M1", "M2", or "VAR".'
                )

        self.c_opt = CCN_Langevin(
            bounds, n_particles, iter, dt, beta, alpha, gamma, lambda_, delta, moment
        )
        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
