/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"
#include "optimizers/particles/Langevin.hh"

class CN_Langevin : public Optimizer
{
public:
  CN_Langevin(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      int k,
      double beta,
      double alpha) : Optimizer(bounds, "CN_Langevin"),
                      base_opt(bounds, n_particles, iter, dt, k, beta, alpha, 0)
  {
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector x)> f);

private:
  Langevin base_opt;
};