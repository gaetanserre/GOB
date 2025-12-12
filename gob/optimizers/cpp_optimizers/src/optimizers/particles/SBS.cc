/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

Eigen::MatrixXd SBS::rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf_matrix)
{
  *rbf_matrix = rbf(particles, this->sigma);
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf_matrix->colwise().sum().transpose().array()) - (*rbf_matrix * particles).array();
  return dxkxy;
}

dynamic SBS::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals, const int &time)
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
  Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(particles.rows(), particles.cols());
  return {((kernel * grads + kernel_grad) / particles.rows()), noise};
}