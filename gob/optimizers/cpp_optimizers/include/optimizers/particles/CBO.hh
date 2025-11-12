/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class CBO : public Particles_Optimizer
{
public:
  CBO(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double lambda,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, dt, batch_size, new LinearScheduler(&this->dt, alpha), "CBO")
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