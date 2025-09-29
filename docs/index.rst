.. GOB documentation master file, created by
   sphinx-quickstart on Tue Sep  9 16:03:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GOB documentation
=================

Welcome to the **GOB** documentation.

**GOB** is a Python package for **global optimization**, providing several algorithms such as SBS, AdaRankOpt, DIRECT, CMA-ES, BayesOpt, and more.

This documentation includes:

- A brief introduction to the package
- A complete API reference

GOB as a Benchmark suite
------------------------
.. code-block:: python

  from gob import GOB
  from gob.benchmarks import PyGKLS, create_bounds

  opt = {
      "AdaRankOpt": {
          "n_eval": 1000,
          "max_trials": 800,
          "max_degree": 80,
      },
      "SBS": {"n_particles": 200, "svgd_iter": 100},
      "Proportion": {"p": 0.9},
  }
  pygkls = PyGKLS(2, 5, [-5, 5], -20, smoothness="D", gen=42)
  gob = GOB(
      ["AdaRankOpt", "AdaLIPO+TR"],
      ["Levy"],
      ["Proportion"],
      bounds=create_bounds(2, -10, 10, 2),
      options=opt,
  )
  gob.run(n_runs=5, verbose=True)

Example output:

.. code-block:: console

  Done for AdaRankOpt on Levy.
  Done for AdaLIPO+TR on Levy.

  Results for Approx:
  +------------+-----------------+
  | Optimizer  |       Levy      |
  +------------+-----------------+
  | AdaRankOpt | 0.0000 ± 0.0000 |
  | AdaLIPO+TR | 0.0000 ± 0.0000 |
  +------------+-----------------+
  Results for Proportion:
  +------------+--------+
  | Optimizer  |  Levy  |
  +------------+--------+
  | AdaRankOpt | 1.0000 |
  | AdaLIPO+TR | 1.0000 |
  +------------+--------+
  Competitive ratios:
  +------------+-------------------+
  | Optimizer  | Competitive ratio |
  +------------+-------------------+
  | AdaRankOpt |      100.0000     |
  | AdaLIPO+TR |       1.0000      |
  +------------+-------------------+

GOB as a library of optimizers
------------------------------

.. code-block:: python

  from gob.benchmarks import *
  from gob.optimizers import *
  from gob import create_bounds

  f = Square()
  bounds = create_bounds(2, -10, 10)
  opt = SBS(bounds)
  res = opt.minimize(f)
  print(f"Results for {opt}: {res[1]}")

Example output:

.. code-block:: console

  Results for SBS: 5.147176623870874e-17

For more details, explore the modules below and refer to the examples for practical guidance.

.. toctree::
   :maxdepth: 2
   
   source/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Optimizers

   source/optimizers/gob.prs
   source/optimizers/gob.gd
   source/optimizers/gob.direct
   source/optimizers/gob.mlsl
   source/optimizers/gob.crs
   source/optimizers/gob.bayesopt
   source/optimizers/gob.cma_es
   source/optimizers/gob.ecp
   source/optimizers/gob.adalipo
   source/optimizers/gob.adarankopt
   source/optimizers/gob.cbo
   source/optimizers/gob.sbs

.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   source/benchmarks/gob.square
   source/benchmarks/gob.ackley
   source/benchmarks/gob.deb
   source/benchmarks/gob.levy
   source/benchmarks/gob.michalewicz
   source/benchmarks/gob.rastrigin
   source/benchmarks/gob.rosenbrock
   source/benchmarks/gob.pygkls

.. toctree::
   :maxdepth: 1
   :caption: Metrics

   source/metrics/gob.ftarget
   source/metrics/gob.proportion