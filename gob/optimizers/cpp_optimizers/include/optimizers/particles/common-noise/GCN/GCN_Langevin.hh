/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Langevin.hh"

class GCN_Langevin : public Optimizer
{
public:
  GCN_Langevin(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      double sigma_cn,
      bool independent_noise = true) : Optimizer(bounds, "GCN-Langevin"),
                                       base_opt(bounds, n_particles, iter, dt, beta, 0)
  {
    this->sigma = sigma_cn;
    this->independent_noise = independent_noise;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  Langevin base_opt;
  double sigma;
  bool independent_noise = true;
};