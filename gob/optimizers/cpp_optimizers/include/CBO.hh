/*
 * Created in 2025 by Gaëtan Serré
 */

#include "particles_optimizer.hh"

class CBO : public Particles_Optimizer
{
public:
  CBO(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      double lambda = 1e-1,
      double epsilon = 1e-2,
      double alpha = 500,
      double sigma = 5) : Particles_Optimizer(bounds, n_particles, iter, false, 0)
  {
    this->lambda = lambda;
    this->epsilon = epsilon;
    this->alpha = alpha;
    this->sigma = sigma;
  };

  virtual Eigen::MatrixXd dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals);

  double lambda;
  double epsilon;
  double alpha;
  double sigma;

private:
  dyn_vector weights(Eigen::MatrixXd &particles, function<double(dyn_vector x)> f, vector<double> *evals);
};