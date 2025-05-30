from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds


# opt.set_stop_criterion(-10)

pygkls = PyGKLS(2, 15, [-5, 5], -100, smoothness="ND", deterministic=True)

f = Square()

bounds = create_bounds(2, -10, 10)

opt = CBO(bounds, beta=5)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = SBS(bounds)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = AdaLIPO_P(bounds)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")
