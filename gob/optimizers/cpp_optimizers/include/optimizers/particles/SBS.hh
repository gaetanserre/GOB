/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class SBS : public Particles_Optimizer
{
public:
  SBS(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      int k = 10000,
      double sigma = 1e-2,
      double dt = 0.01,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, 0.99))
  {
    this->k = k;
    this->sigma = sigma;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  int k;
  double sigma;
  Eigen::MatrixXd rbf(const Eigen::MatrixXd &particles);
  Eigen::MatrixXd rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf);
};