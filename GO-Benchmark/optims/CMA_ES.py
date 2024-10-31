#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer

import numpy as np
import cma


class CMA_ES(Optimizer):
    def __init__(self, domain, m_0, max_evals, sigma0=1):
        self.domain = domain
        self.m_0 = m_0
        self.sigma0 = sigma0
        self.max_evals = max_evals

        # self.optimizer = cma(function, m_0, sigma0)

        """ self.create_optimizer = lambda function: cma(
            f=self.transform_function(function),
            pbounds=self.transform_domain(domain),
            verbose=0,
        ) """

    @staticmethod
    def transform_domain(domain):
        lo = [0] * len(domain)
        up = [0] * len(domain)
        for i, bounds in enumerate(domain):
            lo[i] = bounds[0]
            up[i] = bounds[1]
        return [lo, up]

    """ @staticmethod
    def transform_function(function):
        def intermediate_fun(**params):
            p_values = params.values()
            l = [0] * len(p_values)
            for i, v in enumerate(p_values):
                l[i] = v

            return -function(np.array(l))

        return intermediate_fun """

    def optimize(self, function, verbose=False):
        res = cma.fmin(
            function,
            self.m_0,
            self.sigma0,
            {
                "bounds": self.transform_domain(self.domain),
                "verbose": -9,
                "maxiter": self.max_evals,
            },
        )

        x_opt = res[0]
        f_opt = res[1]

        return (x_opt, f_opt), np.array([]), np.array([])

    def optimize_stats(self, function, verbose=False):
        res = cma.fmin(
            function,
            self.m_0,
            self.sigma0,
            {
                "bounds": self.transform_domain(self.domain),
                "verbose": -9,
                "maxiter": self.max_evals,
            },
        )

        return res[1], res[5], res[6]
