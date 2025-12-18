/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/CBO.hh"

class GCN_CBO : public Optimizer
{
public:
  GCN_CBO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double lambda,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      double sigma_cn) : Optimizer(bounds, "GCN-CBO"),
                         base_opt(bounds, n_particles, iter, dt, lambda, epsilon, beta, sigma, alpha, 0)
  {
    this->sigma = sigma_cn;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  CBO base_opt;
  double sigma;
};