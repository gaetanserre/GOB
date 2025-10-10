/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class PSO : public Particles_Optimizer
{
public:
  PSO(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 1000,
      double dt = 0.01,
      double omega = 0.7,
      double c2 = 2.0,
      double beta = 1e5,
      double alpha = 1,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha))
  {
    this->omega = omega;
    this->c2 = c2;
    this->beta = beta;
    this->velocities = Eigen::MatrixXd::Zero(n_particles, bounds.size());
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  double omega;
  double c2;
  double beta;
  Eigen::MatrixXd velocities;
};