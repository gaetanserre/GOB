/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class RKHS : public Particles_Optimizer
{
public:
  RKHS(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 1000,
      double dt = 0.01,
      double beta = 1e5,
      double sigma = 1,
      double epsilon = 0.5,
      double alpha = 1,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, 0, batch_size, new LinearScheduler(&this->dt, alpha))
  {
    this->beta = beta;
    this->sigma = sigma;
    this->epsilon = epsilon;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  double beta;
  double sigma;
  double epsilon;
  Eigen::MatrixXd compute_noise(const Eigen::MatrixXd &particles);
};