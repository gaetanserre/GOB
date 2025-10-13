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

  if __name__ == "__main__":
      pygkls = PyGKLS(2, 15, [-100, 100], -100, smoothness="ND")
      gob = GOB(
          ["CBO", ("SBS", {"iter": 10}), "SBS", "AdaLIPO+TR", "CMA-ES", "PSO"],
          ["Square", pygkls],
          ["Proportion"],
          bounds=create_bounds(2, -99, 99, 2),
      )
      gob.run(n_runs=10, verbose=1)

Example output:

.. code-block:: console

  Done for CBO on Square.
  Done for SBS on Square.
  Done for SBS (2) on Square.
  Done for AdaLIPO+TR on Square.
  Done for CMA-ES on Square.
  Done for PSO on Square.
  Done for CBO on PyGKLS n°1.
  Done for SBS on PyGKLS n°1.
  Done for SBS (2) on PyGKLS n°1.
  Done for AdaLIPO+TR on PyGKLS n°1.
  Done for CMA-ES on PyGKLS n°1.
  Done for PSO on PyGKLS n°1.

  Results for Approx:
  +------------+-----------------+-----------------------+
  | Optimizer  |      Square     |       PyGKLS n°1      |
  +------------+-----------------+-----------------------+
  |    CBO     | 0.0000 ± 0.0000 |   -100.0000 ± 0.0000  |
  |    SBS     | 0.0000 ± 0.0000 |   -89.5836 ± 8.3975   |
  |  SBS (2)   | 0.0000 ± 0.0000 |   -96.4728 ± 4.2577   |
  | AdaLIPO+TR | 0.0000 ± 0.0000 |   -46.3281 ± 37.4058  |
  |   CMA-ES   | 0.0000 ± 0.0000 | 1712.5414 ± 3511.7282 |
  |    PSO     | 0.0000 ± 0.0000 |   -99.9929 ± 0.0140   |
  +------------+-----------------+-----------------------+
  Results for Proportion:
  +------------+--------+------------+
  | Optimizer  | Square | PyGKLS n°1 |
  +------------+--------+------------+
  |    CBO     | 1.0000 |   1.0000   |
  |    SBS     | 1.0000 |   1.0000   |
  |  SBS (2)   | 1.0000 |   1.0000   |
  | AdaLIPO+TR | 1.0000 |   0.7000   |
  |   CMA-ES   | 1.0000 |   0.6000   |
  |    PSO     | 1.0000 |   1.0000   |
  +------------+--------+------------+
  Competitive ratios:
  +------------+-------------------+
  | Optimizer  | Competitive ratio |
  +------------+-------------------+
  |    CBO     |      39.4351      |
  |    SBS     |      50.5591      |
  |  SBS (2)   |      50.5083      |
  | AdaLIPO+TR |      100.0000     |
  |   CMA-ES   |      50.5000      |
  |    PSO     |      100.0000     |
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

   quickstart

.. toctree::
   :maxdepth: 1

   gob

.. toctree::
   :maxdepth: 1

   source/optimizers/index

.. toctree::
   :maxdepth: 1

   source/benchmarks/index

.. toctree::
   :maxdepth: 1

   source/metrics/index