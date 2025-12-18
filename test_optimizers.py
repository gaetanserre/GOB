from gob.benchmarks import *
from gob.optimizers import *


pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND", gen=42)

f = Levy()

bounds = augment_dimensions(f.visual_bounds, 50)  # f.visual_bounds

""" opt = CBO(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """

n_particles = 150
iter = 300
sigma = 1 / n_particles**2
verbose = False

opt = CBO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_CBO(
    bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose, moment="M1"
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_CBO(
    bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose, moment="M2"
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_CBO(
    bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose, moment="VAR"
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_CBO(
    bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose, moment="MVAR"
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

""" opt = SBS_RKHS(bounds=bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """

""" opt = SBS(
    bounds=bounds, n_particles=n_particles, iter=iter, sigma=sigma, verbose=verbose
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = Langevin(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = PSO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CBO(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_Langevin(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="VAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_SBS(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="MVAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = Full_Noise(bounds=bounds, n_particles=n_particles, iter=1, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = SBS_RKHS(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CN_CBO(
    bounds=bounds,
    n_particles=n_particles,
    iter=iter,
    moment="MVAR",
    verbose=verbose,
)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """

""" opt = CN_Langevin(bounds=bounds, n_particles=n_particles, iter=iter, verbose=verbose)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """

""" opt = AdaLIPO_P(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = PSO(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CMA_ES(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """
