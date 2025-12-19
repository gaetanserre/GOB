/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/CBO.hh"

class SMD_CBO : public Optimizer
{
public:
  SMD_CBO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double lambda,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      double gamma,
      double lambda_cn,
      double delta,
      int moment,
      bool independent_noise = true) : Optimizer(bounds, "SMD-CBO"),
                                       base_opt(bounds, n_particles, iter, dt, lambda, epsilon, beta, sigma, alpha, 0)
  {
    this->gamma = gamma;
    this->lambda = lambda_cn;
    this->delta = delta;
    this->moment = moment;
    this->independent_noise = independent_noise;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  CBO base_opt;
  double gamma;
  double lambda;
  double delta;
  int moment;
  bool independent_noise;
};