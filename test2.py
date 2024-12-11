from gob.benchmarks import *
from gob.optimizers import *
from gob import create_bounds

opt = AdaRankOpt(
    create_bounds(2, -5, 5),
    1000,
    bobyqa=True,
    bobyqa_maxfun=50,
    verbose=True,
)

# opt.set_stop_criteria(10)

pygkls = PyGKLS(2, 5, [-5, 5], -20, smoothness="D", deterministic=True)

f = Square()

res = opt.minimize(pygkls)

print(res, pygkls.n)
