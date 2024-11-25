#
# Created in 2024 by Gaëtan Serré
#

from bayes_opt import BayesianOptimization
from .optimizer import Optimizer
import numpy as np


class BayesOpt(Optimizer):
    def __init__(self, bounds, n_eval=100):
        super().__init__("BayesOpt", bounds)
        self.n_eval = n_eval

        self.create_optimizer = lambda function: BayesianOptimization(
            f=self.transform_function(function),
            pbounds=self.transform_bounds(bounds),
            verbose=0,
            allow_duplicate_points=False,
        )

    @staticmethod
    def transform_bounds(domain):
        p_bounds = {}
        for i, bounds in enumerate(domain):
            p_bounds[f"x{i}"] = bounds
        return p_bounds

    @staticmethod
    def transform_function(function):
        def intermediate_fun(**params):
            return -function(np.array(list(params.values())))

        return intermediate_fun

    def minimize(self, f):
        optimizer = self.create_optimizer(f)
        optimizer.maximize(n_iter=self.n_eval)
        x = []
        for v in optimizer.max["params"].values():
            x.append(v)
        return (np.array(x), -optimizer.max["target"])
