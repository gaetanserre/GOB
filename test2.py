from gob.benchmarks import *
from gob.optimizers import *
import numpy as np

opt = CRS(np.array([[-10, 10], [-10, 10]]), 10)

res = opt.minimize(Square())

print(res)

# print(opt.get_best_per_iter())
