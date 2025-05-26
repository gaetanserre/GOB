from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds


# opt.set_stop_criterion(-10)

pygkls = PyGKLS(2, 15, [-5, 5], -100, smoothness="ND", deterministic=True)

f = Square()

res_mean = 0
for i in range(100):
    print(f"Run {i}")
    opt = ECP(create_bounds(2, -5, 5), 300)
    res = opt.minimize(f)
    res_mean += res[1]
print(f"Results for {opt} : {res_mean / 100}")

f.n = 0

res_mean = 0
for _ in range(100):
    opt = AdaLIPO_P(create_bounds(2, -5, 5), 300)
    res = opt.minimize(f)
    res_mean += res[1]
print(f"Results for {opt} : {res_mean / 100}")
