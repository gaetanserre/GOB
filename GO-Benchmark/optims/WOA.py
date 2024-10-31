#
# Created in 2023 by Gaëtan Serré
#

from .__optimizer__ import Optimizer
import numpy as np
from .whale_optimization import WhaleOptimization


class WOA(Optimizer):
    def __init__(self, domain, n_gen, n_sol, a=2, b=0.5):
        self.domain = domain
        self.n_gen = n_gen
        self.n_sol = n_sol
        self.a = a
        self.b = b

    def optimize(self, function, verbose=False):
        optimizer = WhaleOptimization(
            function,
            self.domain,
            self.n_sol,
            self.b,
            self.a,
            self.a / self.n_gen,
            maximize=False,
        )

        for _ in range(self.n_gen):
            optimizer.optimize()

        best_point = optimizer._best_solutions[-1][1]
        best_value = optimizer._best_solutions[-1][0]

        return (best_point, best_value), np.array([]), np.array([])

    def optimize_(self, function):
        optimizer = WhaleOptimization(
            function,
            self.domain,
            self.n_sol,
            self.b,
            self.a,
            self.a / self.n_gen,
            maximize=False,
        )

        for _ in range(self.n_gen):
            optimizer.optimize()

        return optimizer
