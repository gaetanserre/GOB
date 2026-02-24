#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import ECP as C_ECP


class ECP(CPP_Optimizer):
    """
    Interface for the ECP+TR optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_eval : int
        The maximum number of function evaluations.
    epsilon : float
        The initial Lipschitz constant estimate.
    theta_init : float
        The scaling factor for epsilon.
    C : float
        How many candidates to sample before increasing epsilon.
    max_trials : int
        The maximum number of potential candidates sampled at each iteration.
    trust_region_radius : float
        The trust region radius.
    bobyqa_eval : int
        The number of evaluations for the BOBYQA optimizer.
    verbose : bool
        Whether to print information about the optimization
    """

    def __init__(
        self,
        bounds,
        n_eval=50,
        epsilon=1e-2,
        theta_init=1.001,
        C=1000,
        max_trials=10_000_000,
        trust_region_radius=0.1,
        bobyqa_eval=20,
        verbose=False,
    ):
        super().__init__("ECP+TR", bounds, verbose)

        if n_eval < bobyqa_eval:
            bobyqa_eval = n_eval
            n_eval = 1
        else:
            n_eval = n_eval // bobyqa_eval

        self.c_opt = C_ECP(
            bounds,
            n_eval,
            epsilon,
            theta_init,
            C,
            max_trials,
            trust_region_radius,
            bobyqa_eval,
        )
