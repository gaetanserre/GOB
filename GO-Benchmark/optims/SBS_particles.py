#
# Created in 2023 by Gaëtan Serré
#

import numpy as np
from scipy.spatial.distance import pdist, squareform
from .__optimizer__ import Optimizer
from .CMA_ES import CMA_ES
from .WOA import WOA

print_purple = lambda str: print(f"\033[35m" + str + "\033[0m")


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def step(self, grad, params):
        pass

    def update_states(self, mask):
        pass


class Adam(Optimizer):
    def __init__(
        self,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    ):
        super().__init__(lr)
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.state_m = 0
        self.state_v = 0
        self.state_v_max = 0
        self.t = 0

    def step(self, grad, params):
        self.t += 1

        grad = -grad

        self.state_m = self.betas[0] * self.state_m + (1 - self.betas[0]) * grad
        self.state_v = self.betas[1] * self.state_v + (1 - self.betas[1]) * grad**2

        m_hat = self.state_m / (1 - self.betas[0] ** self.t)
        v_hat = self.state_v / (1 - self.betas[1] ** self.t)

        if self.amsgrad:
            self.state_v_max = np.maximum(self.state_v_max, v_hat)
            return self.lr * m_hat / (np.sqrt(self.state_v_max) + self.eps)
        else:
            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def update_states(self, mask):
        self.state_m = self.state_m[mask]
        self.state_v = self.state_v[mask]


class Static_Optimizer(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def step(self, grad, params):
        return params + self.lr * grad


def gradient(f, x, eps=1e-12):
    f_x = f(x)

    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_p = x.copy()
        x_p[i] += eps
        grad[i] = (f(x_p) - f_x) / eps

    # We remove the number of evaluations required to estimate the gradient
    # f.n -= x.shape[0]

    return grad, f_x


def rbf(x, sigma=-1):
    sq_dist = pdist(x)
    pairwise_dists = squareform(sq_dist) ** 2
    if sigma < 0:  # if sigma < 0, using median trick
        sigma = np.median(pairwise_dists) + 1e-10
        sigma = np.sqrt(0.5 * sigma / np.log(x.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / sigma**2 / 2)

    dxkxy = (x * Kxy.sum(axis=1).reshape(-1, 1) - Kxy @ x).reshape(
        x.shape[0], x.shape[1]
    ) / (sigma**2)

    return Kxy, dxkxy


def svgd(x, logprob_grad, kernel):
    Kxy, dxkxy = kernel(x)

    svgd_grad = (Kxy @ logprob_grad + dxkxy) / x.shape[0]
    return svgd_grad


class SBS_particles(Optimizer):
    def __init__(
        self,
        domain,
        n_particles,
        k_iter,
        svgd_iter,
        sigma=lambda N: 1 / N**2,
        distance_q=0.5,  # 0
        value_q=0.3,  # 0
        lr=0.2,
        adam=True,
        warm_start_iter=None,
    ):
        self.domain = domain
        self.n_particles = n_particles
        self.k_iter = k_iter
        self.svgd_iter = svgd_iter
        self.sigma = sigma
        self.distance_q = distance_q
        self.value_q = value_q
        self.lr = lr
        self.adam = adam
        self.warm_start_iter = warm_start_iter

    def initialize_particles(self, function):
        dim = self.domain.shape[0]

        # Run iterations of CMA-ES

        m_0 = np.random.uniform(self.domain[:, 0], self.domain[:, 1])
        cma = CMA_ES(self.domain, m_0, self.warm_start_iter)
        best_cma, mean, std = cma.optimize_stats(function)

        # Run iterations of WOA

        n_gen = max(1, self.warm_start_iter // self.n_particles)
        woa = WOA(self.domain, n_gen, self.n_particles)
        woa = woa.optimize_(function)
        best_woa = woa._best_solutions[-1][0]

        print(f"PF - Best CMA-ES: {best_cma}. Best WOA: {best_woa}.")

        # Initialize particles
        if best_cma < best_woa:
            print_purple("PF - Initializing particles with CMA-ES.")
            x = np.random.normal(mean, std, size=(self.n_particles, dim))
        else:
            print_purple("PF - Initializing particles with WOA.")
            x = woa._sols

        return np.clip(x, self.domain[:, 0], self.domain[:, 1])

    def remove_particles(self, x, x_new, x_values):
        if x_new.shape[0] > 10:
            distance = np.linalg.norm(x - x_new, axis=1)
            dist_quantile = np.quantile(distance, q=self.distance_q)
            dist_mask = distance < dist_quantile

            value_quantile = np.quantile(x_values, q=self.value_q)
            value_mask = x_values > value_quantile

            mask = ~(dist_mask & value_mask)

            if np.all(mask):
                return x_new, np.ones(x_new.shape[0], dtype=bool)
            else:
                return x_new[mask], mask
        else:
            return x_new, np.ones(x_new.shape[0], dtype=bool)

    def optimize(self, function, verbose=False):
        kernel = lambda N: lambda x: rbf(x, sigma=self.sigma(N))

        dim = self.domain.shape[0]

        if self.warm_start_iter is None:
            x = np.random.uniform(
                self.domain[:, 0], self.domain[:, 1], size=(self.n_particles, dim)
            )
        else:
            x = self.initialize_particles(function)

        # random_indices = np.random.choice(x.shape[0], 5)
        # self.paths = [x[random_indices]]

        n_particles = self.n_particles

        all_points = [x.copy()]
        all_evals = []
        for k in self.k_iter:
            optimizer = Adam(lr=self.lr) if self.adam else Static_Optimizer(lr=self.lr)
            for i in range(self.svgd_iter):
                grads = [0] * n_particles
                fs = [0] * n_particles
                for i, xi in enumerate(x):
                    grad, f_xi = gradient(function, xi)
                    grads[i] = -k * grad
                    fs[i] = f_xi
                all_evals.append(fs)
                svgd_grad = svgd(x, np.array(grads), kernel(n_particles))
                x_new = optimizer.step(svgd_grad, x)

                # clamp to domain
                x_new = np.clip(x_new, self.domain[:, 0], self.domain[:, 1])

                x_new, mask = self.remove_particles(x, x_new, all_evals[-1])
                n_particles = x_new.shape[0]
                optimizer.update_states(mask)

                x = x_new

                # self.paths.append(x[random_indices])

                # save all points
                all_points.append(x.copy())

        np_all_points = None
        for i, np_seq in enumerate(all_points):
            if i == 0:
                np_all_points = np_seq
            else:
                np_all_points = np.concatenate((np_all_points, np_seq), axis=0)

        np_all_evals = None
        for i, np_seq in enumerate(all_evals):
            if i == 0:
                np_all_evals = np_seq
            else:
                np_all_evals = np.concatenate((np_all_evals, np_seq), axis=0)

        best_idx = np.argmin(np_all_evals)
        min_eval = np_all_evals[best_idx]
        best_particle = np_all_points[best_idx]
        if verbose:
            print(f"Best particle found: {best_particle}. Eval at f(best): {min_eval}.")

        return (best_particle, min_eval), np_all_points, np_all_evals
