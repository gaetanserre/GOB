/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class SBS : public Particles_Optimizer
{
public:
  SBS(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      int k,
      double sigma,
      double alpha,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha), "SBS")
  {
    this->k = k;
    this->sigma = sigma;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  int k;
  double sigma;
  Eigen::MatrixXd rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf);
};