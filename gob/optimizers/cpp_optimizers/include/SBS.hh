/*
 * Created in 2025 by Gaëtan Serré
 */

#include "particles_optimizer.hh"

class SBS : public Particles_Optimizer
{
public:
  SBS(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      int k = 10000,
      double sigma = 1e-2,
      double lr = 0.5) : Particles_Optimizer(bounds, n_particles, iter, lr)
  {
    this->k = k;
    this->sigma = sigma;
  };

  virtual Eigen::MatrixXd dynamics(const function<double(dyn_vector x)> &f, const int &time, const Eigen::MatrixXd &particles, vector<double> *evals);

  int k;
  double sigma;

private:
  Eigen::MatrixXd rbf(const Eigen::MatrixXd &particles);
  Eigen::MatrixXd rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf);
};