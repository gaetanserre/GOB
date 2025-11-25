/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class SBS_RKHS : public Particles_Optimizer
{
public:
  SBS_RKHS(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      int k,
      double sigma,
      double alpha,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, batch_size, new LinearScheduler(dt, alpha), "SBS_RKHS")
  {
    this->k = k;
    this->sigma = sigma;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals);

private:
  int k;
  double sigma;
  double theta;
  Eigen::MatrixXd rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf);
  Eigen::MatrixXd compute_noise(const Eigen::MatrixXd &particles, const Eigen::MatrixXd &rbf_matrix);
};