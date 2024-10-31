#
# Created in 2024 by Gaëtan Serré
#

from typing import Tuple
import numpy as np
from .__optimizer__ import Optimizer


def gradient(f, x, eps=1e-12):
    f_x = f(x)

    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps

    return grad, f_x


class Langevin(Optimizer):
    def __init__(self, domain, n_particles, n_iter, kappa, init_lr=0.5):
        self.domain = domain
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.kappa = kappa
        self.init_lr = init_lr

    def mala_acceptance(self, step_size, x, x_new, grad, grad_new, un_pi, un_pi_new):
        q_x_xnew = np.exp(
            (-1 / (4 * step_size))
            * np.linalg.norm(x - x_new - step_size * grad, axis=1) ** 2
        )
        q_xnew_x = np.exp(
            (-1 / (4 * step_size))
            * np.linalg.norm(x_new - x - step_size * grad_new, axis=1) ** 2
        )

        alpha = np.minimum(
            np.ones(x.shape[0]), (un_pi_new * q_x_xnew) / ((un_pi * q_xnew_x) + 1e-10)
        )
        return alpha

    def optimize(self, function, verbose=False):

        x_dim = self.domain.shape[0]
        xi = np.random.normal(0, 1, (self.n_particles, x_dim))
        samples = []
        values = []
        for i in range(self.n_iter):
            grads, fs = [], []
            for x_ in xi:
                grad, f = gradient(function, x_)
                grads.append(-self.kappa * grad)
                fs.append(f)
            step_size = self.init_lr / (i + 1) ** 2

            xi_new = (
                xi
                + step_size * np.array(grads)
                + np.sqrt(2 * step_size)
                * np.random.normal(0, 1, (self.n_particles, x_dim))
            )

            new_grads, new_fs = [], []
            for x_ in xi_new:
                grad, f = gradient(function, x_)
                new_grads.append(-self.kappa * grad)
                new_fs.append(f)

            alpha = self.mala_acceptance(
                step_size,
                xi,
                xi_new,
                np.array(grads),
                np.array(new_grads),
                np.exp(-self.kappa * np.array(fs)),
                np.exp(-self.kappa * np.array(new_fs)),
            )

            u = np.random.uniform(0, 1, self.n_particles)
            mask = u <= alpha
            xi = xi_new * mask.reshape(-1, 1) + xi * (~mask).reshape(-1, 1)

            xi = np.clip(xi_new, self.domain[:, 0], self.domain[:, 1])

            samples += list(xi)
            values += fs
        samples = np.array(samples)
        values = np.array(values)
        best_idx = np.argmin(values)
        return (samples[best_idx], values[best_idx]), samples, values
