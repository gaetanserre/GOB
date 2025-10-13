#
# Created in 2024 by Gaëtan Serré
#

from gob import GOB
from gob.benchmarks import PyGKLS, create_bounds

if __name__ == "__main__":
    pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND")
    gob = GOB(
        ["CBO", ("SBS", {"iter": 10}), "SBS", "AdaLIPO+TR", "CMA-ES", "PSO"],
        ["Square", pygkls],
        ["Proportion"],
        bounds=create_bounds(2, -99, 99, 2),
    )
    gob.run(n_runs=10, verbose=1)
