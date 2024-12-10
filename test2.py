from gob.benchmarks import *
from gob.optimizers import *
import numpy as np

opt = MLSL(np.array([[-1, 1], [-1, 1]]), 1000)

# opt.set_stop_criteria(0.01)

f = Square()

res = opt.minimize(f)

print(res, f.n)

# print(opt.get_best_per_iter())
