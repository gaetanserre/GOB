from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds

opt = AdaRankOpt(
    create_bounds(2, -5, 5),
    1000,
    trust_region_radius=1e-3,
    bobyqa_eval=50,
    verbose=True,
)

# opt.set_stop_criterion(-10)

pygkls = PyGKLS(2, 15, [-5, 5], -100, smoothness="D", deterministic=True)

f = pygkls  # Square()

res = opt.minimize(f)

print(res, f.n, f.min)

f.n = 0

opt = CMA_ES(
    create_bounds(2, -5, 5),
    1000,
)

res = opt.minimize(f)

print(res, f.n, f.min)
