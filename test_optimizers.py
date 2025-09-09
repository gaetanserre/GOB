from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds


pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND", gen=42)

f = pygkls

bounds = create_bounds(2, -99, 99)

opt = CBO(bounds)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = SBS(bounds)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = AdaLIPO_P(bounds, max_trials=50000, bobyqa_eval=20)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = AdaRankOpt(bounds, max_trials=50000, bobyqa_eval=20)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")

opt = ECP(bounds, max_trials=50000, bobyqa_eval=20)
res = opt.minimize(f)
print(f"Results for {opt} : {res[1]}")