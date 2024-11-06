#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
import scipy.special as sp
from scipy.optimize import linprog
from collections import deque
import warnings
import time

from .optimizer import Optimizer


def Bernoulli(p: float):
    """
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)

    Parameters
    ----------
    p : float
        Probability of success.

    Returns
    -------
    bool
        The outcome of the Bernoulli trial.
    """
    a = np.random.uniform(0, 1)
    return a <= p


class AdaRankOpt(Optimizer):
    def __init__(self, bounds, n_eval=100, window_slope=5, max_slope=np.inf):
        super().__init__("AdaRankOpt", bounds)
        self.n_eval = n_eval
        self.window_slope = window_slope
        self.max_slope = max_slope

    @staticmethod
    def polynomial_map(x, k):
        """
        Mapping from R^d to the polynomial feature space of degree k.

        Parameters
        ----------
        x : np.ndarray
            The input vector.
        k : int
            The degree of the polynomial.

        Returns
        -------
        np.ndarray
            The mapped vector.
        """
        start = time.time()
        sym = [f"{i}" for i in range(x.shape[0])]

        def aux(k):
            if k == 1:
                return sym
            else:
                phi_k_m_1 = aux(k - 1)
                x_k = phi_k_m_1.copy()
                phi_k_m_1_only = []
                for e in phi_k_m_1:
                    if len(e.split(" ")) == k - 1:
                        phi_k_m_1_only.append(e)
                for a in sym:
                    phi_k_m_1_only_c = phi_k_m_1_only.copy()
                    for b in phi_k_m_1_only:
                        x_k.append(f"{a} {b}")
                        if a in b:
                            phi_k_m_1_only_c.remove(b)
                    phi_k_m_1_only = phi_k_m_1_only_c
                return x_k

        operations = aux(k)
        assert len(operations) == sp.comb(x.shape[0] + k, x.shape[0]) - 1
        res = np.ones(len(operations))
        for i, op in enumerate(operations):
            splt = op.split(" ")
            for s in splt:
                res[i] *= x[int(s)]
        return res, time.time() - start

    @staticmethod
    def polynomial_matrix(X, k):
        """
        Constructs the polynomial matrix of degree k from the input matrix X, using the polynomial_map function.

        Parameters
        ----------
        X : np.ndarray
            The input matrix.
        k : int
            The degree of the polynomial.

        Returns
        -------
        np.ndarray
            The polynomial matrix.
        """
        d = X.shape[1]
        n = X.shape[0] - 1
        M = np.zeros((n, int(sp.comb(d + k, d) - 1)))
        t = 0
        for i in range(n):
            pm1, t1 = AdaRankOpt.polynomial_map(X[i], k)
            pm2, t2 = AdaRankOpt.polynomial_map(X[i + 1], k)
            M[i] = pm2 - pm1
            t += t1 + t2
        return M, t

    @staticmethod
    def is_polyhedral_set_empty(X, k):
        """
        Checks if the polyhedral set constructed with the polynomial_matrix function is empty using linear programming.

        Parameters
        ----------
        X : np.ndarray
            The input matrix.
        k : int
            The degree of the polynomial.

        Returns
        -------
        np.ndarray
            The polyhedral set.
        """
        M, t = AdaRankOpt.polynomial_matrix(X, k)
        n = M.T.shape[1]
        M = np.vstack((M.T, np.ones(n)))
        b_eq = np.zeros(M.shape[0])
        b_eq[-1] = 1
        """ print("M", M)
        print("M.shape", M.shape)
        print("b_eq", b_eq) """
        with warnings.catch_warnings(action="ignore"):
            start = time.time()
            res = linprog(
                np.ones(n),
                A_eq=M,
                b_eq=b_eq,
                method="simplex",
                options={"maxiter": 10},
            )
            t2 = time.time() - start
        return res.status == 2, t, t2

    def slope_stop_condition(self, last_nb_samples):
        """
        Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve
        is greater than max_slope.
        """
        slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
        return slope > self.max_slope

    def minimize(self, f):
        def p(t):
            """
            Probability of success for exploration/exploitation.
            """
            if t == 1:
                return 1
            else:
                return 1 / np.log(t)

        start = time.time()
        t = 0
        t2 = 0
        k = 1
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        samples = [(x, -f(x))]
        nb_samples = 1
        last_nb_samples = deque([1], maxlen=self.window_slope)

        for i in range(1, self.n_eval + 1):
            if Bernoulli(p(i)):
                x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                samples.append((x, -f(x)))
                samples.sort(key=lambda x: x[1])
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
            else:
                while True:
                    x_t_1 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    fx_t_1 = samples[-1][1] + 1
                    samples.append((x_t_1, fx_t_1))
                    nb_samples += 1
                    last_nb_samples[-1] = nb_samples
                    X = np.array([s[0] for s in samples])
                    b, t_, t2_ = self.is_polyhedral_set_empty(X, k)
                    t += t_
                    t2 += t2_
                    if b:
                        samples[-1] = (x_t_1, -f(x_t_1))
                        samples.sort(key=lambda x: x[1])
                        break
                    elif self.slope_stop_condition(last_nb_samples):
                        print("Polynomials (%):", t / (time.time() - start))
                        print("Simplex (%):", t2 / (time.time() - start))
                        return -samples[-2][1]
                    else:
                        samples.pop()

            while True:
                X = np.array([s[0] for s in samples])
                nb_samples += 1
                last_nb_samples[-1] = nb_samples
                b, t_, t2_ = self.is_polyhedral_set_empty(X, k)
                t += t_
                t2 += t2_
                if b:
                    break
                elif self.slope_stop_condition(last_nb_samples):
                    print("Polynomials (%):", t / (time.time() - start))
                    print("Simplex (%):", t2 / (time.time() - start))
                    return -samples[-1][1]
                k += 1
            last_nb_samples.append(0)
        print("Polynomials (%):", t / (time.time() - start))
        print("Simplex (%):", t2 / (time.time() - start))
        return -samples[-1][1]
