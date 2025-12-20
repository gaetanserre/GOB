/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/PSO.hh"

class SMD_PSO : public Optimizer
{
public:
  SMD_PSO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double omega,
      double c2,
      double beta,
      double alpha,
      double gamma,
      double lambda_cn,
      double delta,
      int moment) : Optimizer(bounds, "SMD-PSO"),
                    base_opt(bounds, n_particles, iter, dt, omega, c2, beta, alpha, 0)
  {
    this->gamma = gamma;
    this->lambda = lambda_cn;
    this->delta = delta;
    this->moment = moment;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  PSO base_opt;
  double gamma;
  double lambda;
  double delta;
  int moment;
  bool independent_noise;
};