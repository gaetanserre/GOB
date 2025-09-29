#
# Created in 2024 by Gaëtan Serré
#

from .benchmark import Benchmark

from gkls import GKLS
import numpy as np


class PyGKLS(Benchmark):
    def __init__(
        self,
        dim,
        num_minima,
        domain,
        global_min,
        global_dist=None,
        global_radius=None,
        gen=None,
        smoothness="D",
    ):
        super().__init__("PyGKLS", global_min)

        self.gkls_function = GKLS(
            dim,
            num_minima,
            domain,
            global_min,
            global_dist,
            global_radius,
            gen,
        )

        self.smoothness = smoothness

        match smoothness:
            case "D":
                self.f = self.gkls_function.get_d_f
                self.gradient = lambda x: (
                    np.array(self.gkls_function.get_d_grad(x)),
                    self.f(x),
                )
            case "D2":
                self.f = self.gkls_function.get_d2_f
                self.gradient = lambda x: (
                    np.array(self.gkls_function.get_d2_grad(x)),
                    self.f(x),
                )
            case "ND":
                self.f = self.gkls_function.get_nd_f

    def expr(self, x):
        return self.f(x)

    def gradient(self, x):
        match self.smoothness:
            case "D":
                return self.gkls_function.get_d_grad(x), self(x)
            case "D2":
                return self.gkls_function.get_d2_grad(x), self(x)
            case "ND":
                raise NotImplementedError("ND GKLS functions are not differentiable")
