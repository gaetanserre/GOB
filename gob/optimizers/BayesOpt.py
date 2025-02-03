#
# Created in 2024 by Gaëtan Serré
#

from bayes_opt import BayesianOptimization
from .optimizer import Optimizer
import numpy as np


class BayesOpt(Optimizer):
    def __init__(self, bounds, n_eval=100):
        assert n_eval >= 2, "The number of evaluations must be at least 2."
        super().__init__("BayesOpt", bounds)
        self.n_eval = n_eval

        self.create_optimizer = lambda function: BayesianOptimization(
            f=self.transform_function(function),
            pbounds=self.transform_bounds(bounds),
            verbose=0,
            allow_duplicate_points=False,
        )

        self.stop_criterion = None

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

    def set_stop_criterion(self, stop_criterion):
        self.stop_criterion = stop_criterion

    def minimize(self, f):
        init_points = min(5, self.n_eval)
        optimizer = self.create_optimizer(f)
        if self.stop_criterion is not None:
            optimizer.maximize(init_points=init_points, n_iter=1)
            for _ in range(self.n_eval - init_points - 1):
                if -optimizer.max["target"] <= self.stop_criterion:
                    break
                optimizer.maximize(init_points=0, n_iter=1)
        else:
            optimizer.maximize(
                init_points=init_points, n_iter=self.n_eval - init_points
            )
        x = []
        for v in optimizer.max["params"].values():
            x.append(v)
        return (np.array(x), -optimizer.max["target"])
