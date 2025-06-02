## Global Optimization Benchmark (GOB)
GOB is a collection of global optimization algorithms implemented in C++ and linked with Python. It also includes a set of analytical benchmark functions and a random function generator ([PyGKLS](https://github.com/gaetanserre/pyGKLS)) to test the performance of these algorithms.

[![CI](https://github.com/gaetanserre/GOB/actions/workflows/build.yml/badge.svg)](https://github.com/gaetanserre/GOB/actions/workflows/build.yml)

### Algorithms
- [AdaLIPO+](https://dl.acm.org/doi/full/10.1145/3688671.3688763)
- [AdaRankOpt](https://arxiv.org/abs/1603.04381)
- [Bayesian Optimization](https://github.com/bayesian-optimization/BayesianOptimization)
- [CMA-ES](https://github.com/CMA-ES/libcmaes)
- [Controlled Random Search](http://dx.doi.org/10.1007/BF00933504)
- [DIRECT](http://dx.doi.org/10.1007/0-306-48332-7_93)
- [Every Call is Precious](https://arxiv.org/abs/2502.04290?)
- [Multi-Level Single-Linkage](https://ageconsearch.umn.edu/record/272327)
- [Stein Boltzmann Sampling](https://arxiv.org/abs/2402.04689)
- [Consensus Based Sampling](https://arxiv.org/abs/1909.09249)
- Gradient Descent
- Pure Random Search

### Installation (Python>=3.10)
Download the corresponding wheel file from the [releases](https://github.com/gaetanserre/GOB/releases) and install it with pip:
```bash
pip install gob-<version>-<architecture>.whl
```

### Usage
This package can be used to design a complete benchmarking framework for global optimization algorithms, testing multiple algorithms on a set of benchmark functions. See [`test_gob.py`](tests/test_gob_tools.py) for an example of how to use it.

The global optimization algorithms can also be used independently. For example, to run the AdaLIPO+ algorithm on a benchmark function:

```python
from gob.optimizers import AdaLIPO_P
from gob import create_bounds

f = lambda x: return x.T @ x

opt = AdaLIPO_P(create_bounds(2, -5, 5), 300)
res = opt.minimize(f)
print(f"Optimal point: {res[0]}, Optimal value: {res[1]}")
```
See [`test_optimizers.py`](tests/test_optimizers.py) for more examples of how to use the algorithms.

### References
- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
- [libcames](https://github.com/CMA-ES/libcmaes)
- [nlopt-python](https://github.com/DanielBok/nlopt-python)

### License
This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See the [LICENSE](LICENSE) file for details.