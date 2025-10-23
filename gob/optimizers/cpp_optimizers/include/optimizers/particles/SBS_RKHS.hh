/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class SBS_RKHS : public Particles_Optimizer
{
public:
  SBS_RKHS(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      double dt = 0.01,
      int k = 10000,
      PyObject *sigma = nullptr,
      double alpha = 0.99,
      double theta = 1,
      double common_noise_sigma = 0,
      int batch_size = 0) : Particles_Optimizer(bounds, n_particles, iter, dt, common_noise_sigma, batch_size, new LinearScheduler(&this->dt, alpha))
  {
    this->k = k;
    this->sigma = sigma;
    this->theta = theta;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  int k;
  PyObject *sigma;
  double theta;
  Eigen::MatrixXd rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf);
  Eigen::MatrixXd compute_noise(const Eigen::MatrixXd &particles, const Eigen::MatrixXd &rbf_matrix);
  double eval_sigma();
};