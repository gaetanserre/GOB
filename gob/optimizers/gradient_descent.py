#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .optimizer import Optimizer


class GD(Optimizer):
    """
    The gradient descent optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_step : int
        The number of steps to take.
    step_size : float
        The step size of the gradient
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(self, bounds, n_step=1000, step_size=1e-3, verbose=False):
        super().__init__("GD", bounds)
        self.n_step = n_step
        self.step_size = step_size
        self.verbose = verbose

    def minimize(self, f):
        d = len(self.bounds)
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(d))
        for i in range(self.n_step):
            grad, f_x = f.gradient(x)
            x -= self.step_size * grad
            print(f"{self.name} eval #{i} : {f_x}")
        return x, f(x)
