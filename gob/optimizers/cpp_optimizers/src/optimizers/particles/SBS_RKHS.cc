/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS_RKHS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

Eigen::MatrixXd SBS_RKHS::rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf_matrix)
{
  *rbf_matrix = rbf(particles, this->sigma);
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf_matrix->colwise().sum().transpose().array()) - (*rbf_matrix * particles).array();
  return dxkxy;
}

Eigen::MatrixXd SBS_RKHS::compute_noise(const Eigen::MatrixXd &particles, const Eigen::MatrixXd &rbf_matrix)
{
  int d = particles.cols();

  // Compute the square root of the RBF matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(rbf_matrix);

  // Ensure positive semi-definiteness
  auto D = es.eigenvalues().cwiseMax(0);
  Eigen::MatrixXd K_sqrt = es.eigenvectors() * D.cwiseSqrt().asDiagonal() * es.eigenvectors().transpose();
  Eigen::MatrixXd K_sqrt_kron = Eigen::kroneckerProduct(K_sqrt, Eigen::MatrixXd::Identity(d, d));

  dyn_vector alphas_tmp = (K_sqrt_kron / particles.rows()) * normal_random_vector(this->re, K_sqrt_kron.rows(), 0, 1);
  Eigen::MatrixXd alphas = Eigen::Map<Eigen::MatrixXd>(alphas_tmp.data(), particles.rows(), d);
  return alphas;
}

dynamic SBS_RKHS::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{

  Eigen::MatrixXd grads(particles.rows(), particles.cols());
  for (int j = 0; j < particles.rows(); j++)
  {
    double f_x;
    grads.row(j) = -gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  Eigen::MatrixXd kernel;
  Eigen::MatrixXd kernel_grad = this->rbf_grad(particles, &kernel);
  Eigen::MatrixXd noise = this->compute_noise(particles, kernel);

  for (int i = 0; i < particles.rows(); i++)
  {
    double eval = f(particles.row(i));
    (*evals)[i] = eval;
  }
  return {((kernel * grads + kernel_grad) / particles.rows()), noise};
}