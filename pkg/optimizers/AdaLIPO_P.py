#
# Created in 2024 by Gaëtan Serré
#

from .optimizer import Optimizer
import numpy as np
from collections import deque


def Bernoulli(p: float):
    """
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)

    Parameters
    ----------
    p : float
        Probability of success.
    """
    a = np.random.uniform(0, 1)
    if a <= p:
        return 1
    else:
        return 0


class AdaLIPO_P(Optimizer):
    def __init__(self, bounds, n_eval=1000, window_slope=5, max_slope=600):
        super().__init__("AdaLIPO_E", bounds)
        self.n_eval = n_eval
        self.window_slope = window_slope
        self.max_slope = max_slope

    @staticmethod
    def slope_stop_condition(last_nb_samples, max_slope):
        """
        Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve
        is greater than max_slope.
        """
        slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
        return slope > max_slope

    def minimize(self, f):
        t = 1
        alpha = 10e-2
        k_hat = 0

        X_1 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        nb_samples = 1

        # We keep track of the last `size_slope` values of nb_samples to compute the slope
        last_nb_samples = deque([1], maxlen=self.window_slope)

        points = np.zeros((self.n_eval, X_1.shape[0]))
        values = np.zeros(self.n_eval)
        points[0] = X_1
        values[0] = -f(X_1)

        def k(i):
            """
            Series of potential Lipschitz constants.
            """
            return (1 + alpha) ** i

        def p(t):
            """
            Probability of success for exploration/exploitation.
            """
            if t == 1:
                return 1
            else:
                return 1 / np.log(t)

        def condition(x, values, k, points, iter):
            """
            Subfunction to check the condition in the loop, depending on the set of values we already have.
            values: set of values of the function we explored (numpy array)
            x: point to check (numpy array)
            k: Lipschitz constant (float)
            points: set of points we have explored (numpy array)
            """
            max_val = np.max(values[:iter])

            left_min = np.min(
                values[:iter] + k * np.linalg.norm(x - points[:iter], ord=2, axis=1)
            )

            return left_min >= max_val

        # Main loop
        ratios = []
        while t < self.n_eval:
            B_tp1 = Bernoulli(p(t))
            if B_tp1 == 1:
                # Exploration
                X_tp1 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
                points[t] = X_tp1
            else:
                # Exploitation
                while True:
                    X_tp1 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    nb_samples += 1
                    last_nb_samples[-1] = nb_samples
                    if condition(X_tp1, values, k_hat, points, t):
                        points[t] = X_tp1
                        break
                    elif self.slope_stop_condition(last_nb_samples, self.max_slope):
                        return -np.max(values)

            value = -f(X_tp1)
            values[t] = value
            for i in range(t):
                ratios.append(
                    np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
                )  # We add all new ratios to the list.

            i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha)))
            k_hat = k(i_hat)

            t += 1
            last_nb_samples.append(0)

        return -np.max(values)
