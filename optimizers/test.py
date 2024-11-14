import optimizers
import numpy as np

bounds = optimizers.create_rect_bounds(-1, 1, 3)

m0 = np.ones(3)

sigma = 10

opt = optimizers.CMA_ES(bounds, 10000, m0, sigma)

f = lambda x: x.T @ x

print(opt.optimize(f))
