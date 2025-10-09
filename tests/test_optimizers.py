from gob.benchmarks import *
from gob.optimizers import *


pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND", gen=42)

f = Square()

bounds = create_bounds(2, -99, 99)

""" opt = CBO(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """

opt = SBS(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

""" opt = AdaLIPO_P(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = PSO(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}")

opt = CMA_ES(bounds)
res = opt.minimize(f)
print(f"Results for {opt}: {res[1]}") """
