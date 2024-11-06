#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
import scipy.special as sp
from scipy.optimize import linprog
from .optimizer import Optimizer


class AdaRankOpt(Optimizer):
    def __init__(self, bounds, n_eval=1000, p=0.1):
        super().__init__("AdaRankOpt", bounds)
        self.n_eval = n_eval
        self.p = p

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
        return res

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
        for i in range(n):
            M[i] = AdaRankOpt.polynomial_map(X[i + 1], k) - AdaRankOpt.polynomial_map(
                X[i], k
            )
        return M

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
        n = X.shape[0] - 1
        M = AdaRankOpt.polynomial_matrix(X, k).T
        M = np.vstack((M, np.ones(n)))
        b_eq = np.zeros(M.shape[0])
        b_eq[-1] = 1
        print("M", M)
        print("M.shape", M.shape)
        print("b_eq", b_eq)
        res = linprog(np.ones(n), A_eq=M, b_eq=b_eq, options={"maxiter": 10})
        print(res)
        return res.status == 2

    def test(self):
        d = 5
        X = np.random.uniform(-3, 3, (3, d))
        print(self.is_polyhedral_set_empty(X, 10))
