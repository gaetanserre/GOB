## Global Optimization Benchmark (GOB)

[![CI](https://github.com/gaetanserre/GOB/actions/workflows/build.yml/badge.svg)](https://github.com/gaetanserre/GOB/actions/workflows/build.yml)
[![CI](https://github.com/gaetanserre/GOB/actions/workflows/build_doc.yml/badge.svg)](https://github.com/gaetanserre/GOB/actions/workflows/build_doc.yml)
[![PyPI version](https://badge.fury.io/py/GOB.svg)](https://badge.fury.io/py/GOB)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

GOB is a collection of global optimization algorithms implemented in C++ and linked with Python. It also includes a set of analytical benchmark functions and a random function generator ([PyGKLS](https://github.com/gaetanserre/pyGKLS)) to test the performance of these algorithms.

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
- [Consensus Based Optimization](https://arxiv.org/abs/1909.09249)
- [*Social Only* Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968)
- Gradient Descent
- Pure Random Search

### Documentation
The documentation is available at [gaetanserre.fr/GOB](https://gaetanserre.fr/GOB/).

### Installation (Python ≥ 3.10)
Install the package via pip from PyPI:
```bash
pip install gob
```

Alternatively, download the corresponding wheel file from the [releases](https://github.com/gaetanserre/GOB/releases) and install it with pip:
```bash
pip install gob-<version>-<architecture>.whl
```

### Build from source
Make sure you have CMake (≥ 3.28), a c++ compiler, and the eigen3 library installed. Then clone the repository and run:
```bash
pip install . -v
```
It should build the C++ extensions and install the package. You can also build the documentation with:
```bash
cd docs
pip install -r requirements.txt
make html
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

### Contributing
Contributions are welcome! Please see the [CONTRIBUTING](CONTRIBUTING.md) file for guidelines.

### References
- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
- [libcames](https://github.com/CMA-ES/libcmaes)
- [nlopt-python](https://github.com/DanielBok/nlopt-python)

### License
This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See the [LICENSE](LICENSE) file for details.
