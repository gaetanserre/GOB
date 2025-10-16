/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/RKHS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Dense>

Eigen::MatrixXd RKHS::rbf(const Eigen::MatrixXd &particles)
{
  Eigen::MatrixXd pdists = pairwise_dist(particles);
  return (-pdists / (2 * this->sigma * this->sigma)).array().exp();
}

Eigen::MatrixXd RKHS::compute_noise(const Eigen::MatrixXd &particles)
{
  Eigen::MatrixXd K = rbf(particles);
  print_matrix(K);
  int d = this->bounds.size();
  Eigen::MatrixXd K_kron = Eigen::kroneckerProduct(K, Eigen::MatrixXd::Identity(d, d));
  print_matrix(K_kron);
  Eigen::MatrixXd K_inv = K_kron.inverse();
  print_matrix(K_inv);
  print_matrix(K_kron * K_inv);
  return Eigen::MatrixXd::Zero(n_particles, d);
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

  dyn_vector stddev = Eigen::VectorXd::Zero(particles.rows());
  Eigen::MatrixXd noise = zero_noise(particles.rows(), this->bounds.size());
  return {drift, stddev, this->compute_noise(particles)};
}