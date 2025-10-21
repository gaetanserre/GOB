/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/RKHS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Dense>

Eigen::MatrixXd RKHS::compute_noise(const Eigen::MatrixXd &particles)
{
  Eigen::MatrixXd K = rbf(particles, this->sigma);
  int d = particles.cols();
  Eigen::MatrixXd K_tmp = K.llt().solve(Eigen::MatrixXd::Identity(K.rows(), K.cols()));
  Eigen::MatrixXd K_inv = Eigen::kroneckerProduct(K_tmp, Eigen::MatrixXd::Identity(d, d)) / particles.rows();
  dyn_vector alphas_tmp = normal_random_variable(K_inv, &this->re)();
  Eigen::MatrixXd alphas = alphas_tmp.reshaped(particles.rows(), d);
  return K * alphas;
}

dynamic RKHS::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector vf = compute_consensus(particles, f, evals, this->beta);

  int i = 0;

  Eigen::MatrixXd drift(particles.rows(), particles.cols());
  for (int i = 0; i < particles.rows(); i++)
  {
    drift.row(i) = -(particles.row(i) - vf.transpose());
  }

  dyn_vector stddev = Eigen::VectorXd::Ones(particles.rows());
  Eigen::MatrixXd noise = zero_noise(particles.rows(), this->bounds.size());
  return {drift + this->epsilon * this->compute_noise(particles), stddev, noise};
}