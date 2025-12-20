#
# Created in 2025 by Gaëtan Serré
#


from ...optimizer import Optimizer
from .whale_optimization import WhaleOptimization
from .common_noise import *


class WOA(Optimizer):
    """
    Interface for the WOA optimizer.

    """

    def __init__(
        self, bounds, n_particles=200, iter=100, common_noise=None, verbose=False
    ):
        super().__init__("WOA", bounds)
        self.n_particles = n_particles
        self.iter = iter
        self.common_noise = common_noise
        self.verbose = verbose

    def minimize(self, f):
        optimizer = WhaleOptimization(
            f,
            self.bounds,
            self.n_particles,
            0.5,
            2,
            2 / self.iter,
            maximize=False,
        )
        if self.common_noise == "M1":
            for _ in range(self.iter):
                optimizer.optimize()
                optimizer._sols = smd_m1(optimizer._sols, dt=0)
        elif self.common_noise == "M2":
            for _ in range(self.iter):
                optimizer.optimize()
                optimizer._sols = smd_m2(optimizer._sols, delta=2.1, lam_=0, dt=0.5)
        else:
            for _ in range(self.iter):
                optimizer.optimize()

        best_point = optimizer._best_solutions[-1][1]
        best_value = optimizer._best_solutions[-1][0]
        return best_point, best_value
