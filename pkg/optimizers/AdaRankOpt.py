#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from .optimizer import Optimizer


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
    set
        The mapped vector.
    """
    sym = [f"{i}" for i in range(x.shape[0])]

    def up_to_permutation(e, l):
        for i in range(len(l)):
            if e == l[i]:
                return True
            # Check if e is a permutation of l[i]
            if len(e) == len(l[i]) and sorted(e) == sorted(l[i]):
                return True
        return False

    def aux(k):
        if k == 1:
            return sym
        else:
            x_k_m_1 = aux(k - 1)
            x_k_1 = x_k_m_1.copy()
            for a in sym:
                for b in x_k_m_1:
                    if not up_to_permutation(f"{b} {a}", x_k_1):
                        x_k_1.append(f"{a} {b}")
            return x_k_1

    operations = aux(k)
    res = np.ones(len(operations))
    for i, op in enumerate(operations):
        splt = op.split(" ")
        for s in splt:
            res[i] *= x[int(s)]
    return res


class AdaRankOpt(Optimizer):
    def __init__(self, bounds, n_eval=1000, p=0.1):
        super().__init__("AdaRankOpt", bounds)
        self.n_eval = n_eval
        self.p = p
