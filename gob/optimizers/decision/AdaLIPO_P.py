#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import AdaLIPO_P as C_AdaLIPO_P


class AdaLIPO_P(CPP_Optimizer):
    """
    Interface for the AdaLIPO+TR optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_eval : int
        The maximum number of function evaluations.
    max_trials : int
        The maximum number of potential candidates sampled at each iteration.
    trust_region_radius : float
        The trust region radius.
    bobyqa_eval : int
        The number of evaluations for the BOBYQA optimizer.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_eval=1000,
        max_trials=50_000,
        trust_region_radius=0.1,
        bobyqa_eval=20,
        verbose=False,
    ):
        super().__init__("AdaLIPO+TR", bounds, verbose)

        if n_eval < bobyqa_eval:
            bobyqa_eval = n_eval
            n_eval = 1
        else:
            n_eval = n_eval // bobyqa_eval

        self.c_opt = C_AdaLIPO_P(
            bounds,
            n_eval,
            max_trials,
            trust_region_radius,
            bobyqa_eval,
        )
