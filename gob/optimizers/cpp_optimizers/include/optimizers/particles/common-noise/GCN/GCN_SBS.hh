/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS.hh"

class GCN_SBS : public Optimizer
{
public:
  GCN_SBS(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double sigma,
      double sigma_cn) : Optimizer(bounds, "GCN-SBS"),
                         base_opt(bounds, n_particles, iter, dt, sigma, 0)
  {
    this->sigma = sigma_cn;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  SBS base_opt;
  double sigma;
};