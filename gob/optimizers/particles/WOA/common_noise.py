#
# Created in 2025 by Gaëtan Serré
#

import numpy as np


def compute_moment(particles, r, dim):
    moment = 0
    for i in range(particles.shape[0]):
        moment += pow(particles[i, dim], r)
    moment /= particles.shape[0]
    return moment


def smd_m1(particles, dt):
    d = particles.shape[1]
    gaussian_noise = np.random.normal(0, 1, d)
    return particles + np.sqrt(dt) * gaussian_noise


def smd_m2(particles, delta, lam_, dt):
    d = particles.shape[1]
    drift = np.zeros(d)
    noise = np.zeros(d)
    for dim in range(d):
        moment = compute_moment(particles, 2, dim)
        drift[dim] = particles[:, dim] * (delta - 3 / 2) / (4 * pow(lam_ + moment, 2))
        noise[dim] = particles[:, dim] / (2 * (lam_ + moment))
    gaussian_noise = np.random.normal(0, 1, d)
    return particles + drift * dt + np.sqrt(dt) * np.diagonal(noise).dot(gaussian_noise)
