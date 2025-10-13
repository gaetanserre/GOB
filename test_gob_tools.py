#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob.benchmarks import PyGKLS, create_bounds
import inspect
import gob.benchmarks as gb

if __name__ == "__main__":
    pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND")

    benchmarks = []
    bounds = []
    for name, obj in inspect.getmembers(gb, inspect.isclass):
        if name != "PyGKLS":
            benchmarks.append(obj())
            bounds.append(benchmarks[-1].visual_bounds)
    benchmarks.append(pygkls)
    bounds.append(create_bounds(2, -99, 99))

    common_noise_sigma = 300
    gob = GOB(
        [("CBO", {"common_noise_sigma": common_noise_sigma}), "CBO"],
        benchmarks[:3] + ["Square"],
        ["Proportion"],
        bounds=bounds,
    )
    gob.run(n_runs=1, verbose=1)
