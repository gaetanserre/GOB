#
# Created in 2024 by Gaëtan Serré
#

import numpy as np
from optims.CMA_ES import CMA_ES
from optims.WOA import WOA
from optims.Langevin import Langevin
from optims.SBS import SBS
from optims.SBS_particles import SBS_particles

def load_algo(name, bounds, num_evals):
  if name == "CMA":
    m_0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
    return CMA_ES(bounds, m_0, max_evals=num_evals)
  elif name == "WOA":
      return WOA(bounds, n_gen=30, n_sol=num_evals // 30)
  elif name == "Langevin":
      return Langevin(bounds, n_particles=500, n_iter=150, kappa=10_000, init_lr=0.2)
  elif name == "SBS":
      return SBS(
         bounds,
         n_particles=500,
         k_iter=[10_000],
         svgd_iter=300,
         sigma=1 / 500**2,
         lr=0.2
      )
  elif name == "SBS_pf":
      return SBS_particles(
          bounds,
          n_particles=500,
          k_iter=[10_000],
          svgd_iter=300,
          sigma=lambda N: 1 / N**2,
          lr=0.2,
      )
  else:
     raise NotImplementedError(f"{name} not implemented.")