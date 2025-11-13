#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob.benchmarks import PyGKLS, create_bounds, augment_dimensions
import inspect
import gob.benchmarks as gb

n_particles = 150
iter = 300
sigma = 1 / n_particles**2
dim = 10

if __name__ == "__main__":
    pygkls = PyGKLS(dim, 15, [-100, 100], -100, smoothness="ND")

    benchmarks = []
    bounds = []
    for name, obj in inspect.getmembers(gb, inspect.isclass):
        if name != "PyGKLS":
            benchmarks.append(obj())
            bounds.append(augment_dimensions(benchmarks[-1].visual_bounds, dim))
    benchmarks.append(pygkls)
    bounds.append(create_bounds(dim, -99, 99))

    gob = GOB(
        [
            ("Langevin", {"n_particles": n_particles, "iter": iter}),
            ("SBS", {"n_particles": n_particles, "iter": iter, "sigma": sigma}),
        ],
        benchmarks,
        ["Proportion"],
        bounds=bounds,
    )
    gob.run(n_runs=5, verbose=1)
