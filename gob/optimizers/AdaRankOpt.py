#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.special as sp
from tqdm import tqdm
import pyomo.environ as pyo


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
    def __init__(self, bounds, n_eval=1000, method="lstsq"):
        """
        Implementation of the AdaRankOpt algorithm.

        Parameters
        ----------
        bounds : np.ndarray
            The bounds of the search space.
        n_eval : int
            The number of evaluations.
        method : str
            The method used for the linear programming problem. Can be "lstsq" or "simplex".
        """
        super().__init__("AdaRankOpt", bounds)
        self.n_eval = n_eval
        self.method = method
        self.poly_features = PolynomialFeatures(1)

    def polynomial_map(self, x):
        """
        Mapping from R^d to the polynomial feature space of degree k.

        Parameters
        ----------
        x : np.ndarray
            The input vector.

        Returns
        -------
        np.ndarray
            The mapped vector.
        """
        res = self.poly_features.fit_transform(x.reshape(1, -1))[0][1:]
        assert (
            len(res) == sp.comb(x.shape[0] + self.poly_features.degree, x.shape[0]) - 1
        )
        return res

    def polynomial_matrix(self, X):
        """
        Constructs the polynomial matrix of degree k from the input matrix X, using the polynomial_map function.

        Parameters
        ----------
        k : int
            The degree of the polynomial.

        Returns
        -------
        np.ndarray
            The polynomial matrix.
        """
        d = X.shape[1]
        n = X.shape[0] - 1
        M = np.zeros((n, int(sp.comb(d + self.poly_features.degree, d) - 1)))
        for i in range(n):
            M[i] = self.polynomial_map(X[i + 1]) - self.polynomial_map(X[i])
        return M.T

    def is_polyhedral_set_empty(
        self,
        X,
        tol=1e-6,
    ):
        """
        Checks if the polyhedral set constructed with the polynomial_matrix function is empty using linear programming.

        Parameters
        ----------
        X : np.ndarray
            The input matrix.
        tol : float
            The tolerance for the lstsq algorithm.

        Returns
        -------
        np.ndarray
            The polyhedral set.
        """
        M = self.polynomial_matrix(X)
        n = M.shape[1]

        if self.method == "lstsq":
            M = np.vstack((M, np.ones(n)))
            b_eq = np.zeros(M.shape[0])
            b_eq[-1] = 1
            lambdas = np.linalg.lstsq(M, b_eq)[0]
            return np.all(0 <= lambdas) and np.abs(b_eq - M @ lambdas).sum() < tol
        elif self.method == "simplex":
            m = M.shape[0]
            model = pyo.ConcreteModel()
            model.x = pyo.Var(range(n + m), domain=pyo.NonNegativeReals)
            model.obj = pyo.Objective(expr=sum(model.x[i] for i in range(n, n + m)))
            model.constraint = pyo.Constraint(
                expr=sum(model.x[i] for i in range(n)) == 1
            )
            model.I = pyo.RangeSet(0, n - 1)
            model.J = pyo.RangeSet(n, n + m - 1)

            def ax_constraint_rule(m, j):
                return sum(M[j - n, i] * m.x[i] for i in m.I) <= m.x[j]

            model.AxbConstraint = pyo.Constraint(model.J, rule=ax_constraint_rule)
            solver = pyo.SolverFactory("glpk")

            solver.options["tmlim"] = 1
            try:
                res = solver.solve(model, tee=True)
                x = [pyo.value(model.x[i]) for i in model.x]
                return np.sum(x[n:]) >= tol
            except:
                return True

    def minimize(self, f):
        x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        samples = [(x, -f(x))]

        for i in tqdm(range(1, self.n_eval + 1)):
            if Bernoulli(1 / 10):
                x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                samples.append((x, -f(x)))
                samples.sort(key=lambda x: x[1])
            else:
                while True:
                    x_t_1 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    fx_t_1 = samples[-1][1] + 1
                    samples.append((x_t_1, fx_t_1))
                    X = np.array([s[0] for s in samples])
                    if self.is_polyhedral_set_empty(X):
                        samples[-1] = (x_t_1, -f(x_t_1))
                        samples.sort(key=lambda x: x[1])
                        break
                    else:
                        samples.pop()

            while True:
                X = np.array([s[0] for s in samples])
                if self.is_polyhedral_set_empty(X):
                    break
                self.poly_features.degree += 1
        return -samples[-1][1]
