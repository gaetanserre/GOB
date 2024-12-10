#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer

import numpy as np
import nlopt


class MLSL(Optimizer):
    def __init__(self, bounds, n_eval=1000):
        super().__init__("MLSL", bounds)
        self.n_eval = n_eval
        self.opt = nlopt.opt(nlopt.GN_MLSL, len(self.bounds))

    def minimize(self, f):
        def f_(x, grad):
            if grad.size > 0:
                grad[:] = f.gradient(x)
            return f(x)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        self.opt.set_min_objective(f_)
        self.opt.set_lower_bounds(lb)
        self.opt.set_upper_bounds(ub)
        self.opt.set_maxeval(self.n_eval)
        self.opt.set_local_optimizer(nlopt.opt(nlopt.LN_COBYLA, len(self.bounds)))

        x = np.random.uniform(lb, ub)
        best = self.opt.optimize(x)
        return (best, f(best))

    def set_stop_criteria(self, stop_criteria):
        self.opt.set_stopval(stop_criteria)
