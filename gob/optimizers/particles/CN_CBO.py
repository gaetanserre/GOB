#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import CN_CBO as CCN_CBO


class CN_CBO(CPP_Optimizer):
    """
    Interface for the Common noise CBO optimizer.

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
        iter=1000,
        dt=0.01,
        lam=1,
        epsilon=1e-2,
        beta=1,
        sigma=5.1,
        alpha=1,
        gamma=1,
        lambda_=0,
        delta=2.1,
        moment="M2",
        verbose=False,
    ):
        super().__init__("CN-CBO", bounds, verbose)

        match moment:
            case "M1":
                moment = 0
            case "M2":
                moment = 1
            case "VAR":
                moment = 2
            case _:
                raise ValueError(
                    f'Invalid moment type. Choose from "M1", "M2", or "VAR".'
                )

        self.c_opt = CCN_CBO(
            bounds,
            n_particles,
            iter,
            dt,
            lam,
            epsilon,
            beta,
            sigma,
            alpha,
            gamma,
            lambda_,
            delta,
            moment,
        )
