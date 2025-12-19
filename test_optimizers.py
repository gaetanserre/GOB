from gob.benchmarks import *
from gob.optimizers import *


pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND", gen=42)

f = Square()

bounds = augment_dimensions(f.visual_bounds, 2)  # f.visual_bounds

n_particles = 150
iter = 1000
sigma = 1 / n_particles**2
verbose = False

opt = SBS(
    bounds=bounds, n_particles=n_particles, iter=iter, sigma=sigma, verbose=verbose
)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = Langevin(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = PSO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = CBO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = Full_Noise(bounds=bounds, n_particles=n_particles, iter=1, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = SMD_Langevin(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="VAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = SMD_SBS(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="MVAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = SMD_CBO(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="MVAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = GCN_Langevin(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = GCN_SBS(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")

opt = GCN_CBO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt} on {f}: {res[1]}")
