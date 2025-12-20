/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/PSO.hh"

class GCN_PSO : public Optimizer
{
public:
  GCN_PSO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double omega,
      double c2,
      double beta,
      double alpha,
      double sigma_cn) : Optimizer(bounds, "GCN-PSO"),
                         base_opt(bounds, n_particles, iter, dt, omega, c2, beta, alpha, 0)
  {
    this->sigma = sigma_cn;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  PSO base_opt;
  double sigma;
};