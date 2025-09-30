from gob.benchmarks import *
import numpy as np

d= 4
x = np.array([i * (d + 1 - i) for i in range(1, d+1) ])

f = Trid()
print(f(x))