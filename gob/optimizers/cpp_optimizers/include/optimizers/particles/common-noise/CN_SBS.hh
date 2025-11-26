/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS.hh"

class CN_SBS : public Optimizer
{
public:
  CN_SBS(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double sigma,
      double gamma,
      double lambda,
      double delta,
      int moment) : Optimizer(bounds, "CN_SBS"),
                    base_opt(bounds, n_particles, iter, dt, sigma, 0)
  {
    this->gamma = gamma;
    this->lambda = lambda;
    this->delta = delta;
    this->moment = moment;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  SBS base_opt;
  double gamma;
  double lambda;
  double delta;
  int moment;
};