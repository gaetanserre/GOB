.. GOB documentation master file, created by
   sphinx-quickstart on Tue Sep  9 16:03:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GOB documentation
=================

Welcome to the **GOB** documentation.

**GOB** is a Python package for **global optimization**, providing several algorithms such as DIRECT, CMA-ES, BayesOpt, and more.

This documentation includes:

- A brief introduction to the package
- A complete API reference
- Usage examples

Example usage:

.. code-block:: python

  from gob.benchmarks import *
  from gob.optimizers import *
  from gob import create_bounds

  f = Square()
  bounds = create_bounds(2, -10, 10)
  opt = SBS(bounds)
  res = opt.minimize(f)
  print(f"Results for {opt} : {res[1]}")

For more details, explore the modules below and refer to the examples for practical guidance.

.. toctree::
   :hidden:

   source/quickstart
   source/modules

.. toctree::
   :maxdepth: 2
   :caption: Benchmark
   :hidden:

   source/gob.benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Metrics
   :hidden:

   source/gob.metrics

.. toctree::
   :maxdepth: 2
   :caption: Optimizers
   :hidden:

   source/gob.optimizers