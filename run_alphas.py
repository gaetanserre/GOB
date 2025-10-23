from gob import GOB
import inspect
import gob.benchmarks as gb
import numpy as np

if __name__ == "__main__":
    benchmarks = []
    bounds = []
    for name, obj in inspect.getmembers(gb, inspect.isclass):
        if name != "PyGKLS":
            benchmarks.append(obj())
            bounds.append(benchmarks[-1].visual_bounds)

    n_particles = 500
    iter = 300
    sigma = 1 / n_particles**2
    thetas = np.linspace(1 / 4, 2, 6)
    print(f"Thetas: {thetas}")
    optimizers = [
        ("SBS", {"n_particles": n_particles, "iter": iter, "sigma": sigma})
    ] + [
        (
            "SBS-RKHS",
            {"n_particles": n_particles, "iter": iter, "sigma": sigma, "theta": theta},
        )
        for theta in thetas
    ]

    gob = GOB(
        optimizers,
        ["Rastrigin"],
        ["Proportion"],
        bounds=[gb.create_bounds(2, -5.12, 5.12)],
    )
    print(gob.run(n_runs=10, verbose=1, latex_table=True))
