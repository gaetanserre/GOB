import optimizers
import numpy as np


bounds = optimizers.create_rect_bounds(-1, 1, 2)

prs = optimizers.PRSWrapper(bounds, 100)

f = lambda x: np.sum(x)

print(prs.optimize)
