#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer

import numpy as np
import nlopt


class Direct(Optimizer):
    """
    Interface for the Direct optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_eval : int
        The maximum number of function evaluations.
    verbose : bool
        Whether to print information about the optimization
    """

    def __init__(self, bounds, n_eval=1000, verbose=False):
        super().__init__("Direct", bounds)
        self.n_eval = n_eval
        self.verbose = verbose
        self.opt = nlopt.opt(nlopt.GN_DIRECT, len(self.bounds))

    def minimize(self, f):
        def f_(x, grad):
            if grad.size > 0:
                grad[:] = f.gradient(x)
            return f(x)

        if self.verbose:
            f_ = self.verbose_function(f_)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        self.opt.set_min_objective(f_)
        self.opt.set_lower_bounds(lb)
        self.opt.set_upper_bounds(ub)
        self.opt.set_maxeval(self.n_eval)

        x = np.random.uniform(lb, ub)
        best = self.opt.optimize(x)
        return (best, f(best))

    def set_stop_criterion(self, stop_criterion):
        self.opt.set_stopval(stop_criterion)
