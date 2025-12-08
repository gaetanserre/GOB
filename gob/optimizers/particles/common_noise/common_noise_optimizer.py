#
# Created in 2025 by Gaëtan Serré
#

from ...cpp_optimizer import CPP_Optimizer


class CN_Optimizer(CPP_Optimizer):
    """
    Interface for C++ based optimizers.

    Parameters
    ----------
    name : str
        The name of the optimizer.
    bounds : ndarray
        The bounds of the search space.
    moment : str, optional
        The moment to use for the common noise. Can be "M1", "M2", "VAR", "MVAR".
    verbose : bool
        Whether to print information about the optimization process.
    """

    def __init__(self, name, bounds, moment, verbose=False):
        super().__init__(name, bounds)

        match moment:
            case "M1":
                self.moment = 0
            case "M2":
                self.moment = 1
            case "VAR":
                self.moment = 2
            case "MVAR":
                self.moment = 3
            case _:
                raise ValueError(
                    'Invalid moment type. Choose from "M1", "M2", "VAR", or "MVAR".'
                )
