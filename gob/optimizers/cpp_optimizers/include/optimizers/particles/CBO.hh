/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class CBO : public Particles_Optimizer
{
public:
  CBO(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 1000,
      double dt = 0.01,
      double lambda = 1,
      double epsilon = 1e-2,
      double beta = 1,
      double sigma = 5.1,
      double alpha = 1,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha))
  {
    this->lambda = lambda;
    this->epsilon = epsilon;
    this->beta = beta;
    this->sigma = sigma;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  double lambda;
  double epsilon;
  double beta;
  double sigma;
};