/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class PSO : public Particles_Optimizer
{
public:
  PSO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double omega,
      double c2,
      double beta,
      double alpha,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha))
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