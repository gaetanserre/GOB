#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer

import numpy as np
import nlopt


class CRS(Optimizer):
    def __init__(self, bounds, n_eval=1000):
        super().__init__("CRS", bounds)
        self.n_eval = n_eval

    def minimize(self, f):
        def f_(x, grad):
            if grad.size > 0:
                grad[:] = f.gradient(x)
            return f(x)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        opt = nlopt.opt(nlopt.GN_CRS2_LM, len(self.bounds))
        opt.set_min_objective(f_)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_maxeval(self.n_eval)

        x = np.random.uniform(lb, ub)
        best = opt.optimize(x)
        print(opt.get_param())
        return (best, f(best))
