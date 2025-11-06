/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class Langevin : public Particles_Optimizer
{
public:
  Langevin(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      double dt = 0.1,
      int k = 10000,
      double beta = 0.5,
      double alpha = 1,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha))
  {
    this->k = k;
    this->beta = beta;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  int k;
  double beta;
};