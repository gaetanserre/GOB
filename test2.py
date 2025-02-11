from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds

opt = ECP(create_bounds(2, -5, 5), 300)

# opt.set_stop_criterion(-10)

pygkls = PyGKLS(2, 15, [-5, 5], -100, smoothness="ND", deterministic=True)

f = pygkls

res_mean = 0
for _ in range(100):
    res = opt.minimize(f)
    res_mean += res[1]
print(f"Results for {opt} : {res_mean / 100}")


f.n = 0

opt = AdaLIPO_P(
    create_bounds(2, -5, 5),
    300,
)

res_mean = 0
for _ in range(100):
    res = opt.minimize(f)
    res_mean += res[1]
print(f"Results for {opt} : {res_mean / 100}")
