/*
 * Created in 2025 by Gaëtan Serré
 */

#include "particles_optimizer.hh"

class CBO : public Particles_Optimizer
{
public:
  CBO(
      vec_bounds bounds,
      int n_particles = 20,
      int iter = 1000,
      double dt = 0.01,
      double lambda = 1,
      double epsilon = 1e-2,
      double beta = 1,
      double sigma = 5.1,
      bool use_batch = true) : Particles_Optimizer(bounds, n_particles, iter, false, 0)
  {
    this->dt = dt;
    this->lambda = lambda;
    this->epsilon = epsilon;
    this->beta = beta;
    this->sigma = sigma;
    this->use_batch = use_batch;
  };

  virtual Eigen::MatrixXd dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals);
  virtual Eigen::MatrixXd full_dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals);
  virtual Eigen::MatrixXd batch_dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals);

  double dt;
  double lambda;
  double epsilon;
  double beta;
  double sigma;
  bool use_batch;

private:
  dyn_vector weights(Eigen::MatrixXd &particles, function<double(dyn_vector x)> f, vector<double> *evals);
};