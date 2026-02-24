#
# Created in 2024 by Gaëtan Serré
#

<<<<<<<< HEAD:gob/optimizers/particles/SBS.py
from ..optimizer import Optimizer
from ..cpp_optimizers import SBS as C_SBS
========
from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import Full_Noise as C_Full_Noise
import numpy as np
>>>>>>>> sde:gob/optimizers/particles/Full_Noise.py


class Full_Noise(CPP_Optimizer):
    """
    Interface for the Langevin optimizer.

    Parameters
    ----------
    bounds : ndarray
        The bounds of the search space.
    n_particles : int
        The number of particles.
    iter : int
        The number of iterations.
    dt : float
        The time step.
<<<<<<<< HEAD:gob/optimizers/particles/SBS.py
    k : float
        The kappa exponent.
    sigma : float
        The kernel bandwidth.
========
>>>>>>>> sde:gob/optimizers/particles/Full_Noise.py
    alpha : float
        The coefficient to decrease the step size.
    batch_size : int
        The batch size for the mini-batch optimization. If 0, no mini-batch
        optimization is used.
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(
        self,
        bounds,
        n_particles=200,
        iter=100,
<<<<<<<< HEAD:gob/optimizers/particles/SBS.py
        dt=10,
        k=1,
        sigma=0.1,
========
        dt=0.1,
>>>>>>>> sde:gob/optimizers/particles/Full_Noise.py
        alpha=0.99,
        batch_size=0,
        verbose=False,
    ):
        super().__init__("Full-Noise", bounds, verbose)
        self.c_opt = C_Full_Noise(bounds, n_particles, iter, dt, alpha, batch_size)

    def minimize(self, f):
        res = super().minimize(f)
        return (res[0], f(np.array(res[0])))
