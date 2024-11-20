import pyomo.environ as pyo
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.special as sp

poly_features = PolynomialFeatures(2, include_bias=False)


def polynomial_map(x):
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
    res = poly_features.fit_transform(x.reshape(1, -1))[0]
    assert len(res) == sp.comb(x.shape[0] + poly_features.degree, x.shape[0]) - 1
    return res


def polynomial_matrix(X):
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
    M = np.zeros((n, int(sp.comb(d + poly_features.degree, d) - 1)))
    for i in range(n):
        M[i] = polynomial_map(X[i + 1]) - polynomial_map(X[i])
    return M.T


def is_polyhedral_set_empty(
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
    M = polynomial_matrix(X)
    n = M.shape[1]
    m = M.shape[0]
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(n + m), domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=sum(model.x[i] for i in range(n, n + m)))
    model.constraint = pyo.Constraint(expr=sum(model.x[i] for i in range(n)) == 1)
    model.I = pyo.RangeSet(0, n - 1)
    model.J = pyo.RangeSet(n, n + m - 1)

    def ax_constraint_rule(m, j):
        return sum(M[j - n, i] * m.x[i] for i in m.I) <= m.x[j]

    model.AxbConstraint = pyo.Constraint(model.J, rule=ax_constraint_rule)
    model.write("model.lp")
    solver = pyo.SolverFactory("glpk")

    solver.options["tmlim"] = 1
    try:
        res = solver.solve(model, tee=True)
        x = [pyo.value(model.x[i]) for i in model.x]
        return np.sum(x[n:]) >= tol
    except:
        return True


x = np.array([[3, 4], [5, 6], [7, 8]])
print(is_polyhedral_set_empty(x))
