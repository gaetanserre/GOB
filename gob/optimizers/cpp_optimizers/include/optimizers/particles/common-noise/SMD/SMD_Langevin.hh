/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Langevin.hh"

class SMD_Langevin : public Optimizer
{
public:
  SMD_Langevin(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      double gamma,
      double lambda,
      double delta,
      int moment) : Optimizer(bounds, "SMD-Langevin"),
                    base_opt(bounds, n_particles, iter, dt, beta, 0)
  {
    this->gamma = gamma;
    this->lambda = lambda;
    this->delta = delta;
    this->moment = moment;
  }

  virtual void set_stop_criterion(double stop_criterion);
  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  Langevin base_opt;
  double gamma;
  double lambda;
  double delta;
  int moment;
};