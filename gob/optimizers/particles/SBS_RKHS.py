#
# Created in 2024 by Gaëtan Serré
#

from ..cpp_optimizer import CPP_Optimizer
from ..cpp_optimizers import SBS_RKHS as C_SBS_RKHS


class SBS_RKHS(CPP_Optimizer):
    """
    Interface for the SBS RKHS optimizer.

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
    sigma : float
        The kernel bandwidth.
    sigma_noise : float
        The kernel bandwidth for noise.
    theta : float
        The regularization parameter for the RKHS-based noise.
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
        dt=10,
        sigma=0.1,
        sigma_noise=3,
        batch_size=0,
        verbose=False,
    ):
        super().__init__("SBS-RKHS", bounds, verbose)

        self.c_opt = C_SBS_RKHS(
            bounds,
            n_particles,
            iter,
            dt,
            sigma,
            sigma_noise,
            batch_size,
        )
