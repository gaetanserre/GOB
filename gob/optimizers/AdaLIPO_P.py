#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
from .cpp_optimizers import AdaLIPO_P as C_AdaLIPO_P


class AdaLIPO_P(Optimizer):
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
        super().__init__("AdaLIPO+TR", bounds)

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

        self.verbose = verbose

    def minimize(self, f):
        if self.verbose:
            f = self.verbose_function(f)
        return self.c_opt.minimize(f)

    def set_stop_criterion(self, stop_criterion):
        self.c_opt.set_stop_criterion(stop_criterion)

    def __del__(self):
        del self.c_opt
