/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

dyn_vector gradient(dyn_vector x, const function<double(dyn_vector x)> &f, double *f_x, double tol = 1e-9)
{
  dyn_vector grad(x.size());
  *f_x = f(x);
  for (int i = 0; i < x.size(); i++)
  {
    dyn_vector x_plus = x;
    x_plus[i] += tol;
    grad(i) = ((f(x_plus) - *f_x) / tol);
  }
  return grad;
}

Eigen::MatrixXd SBS::rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf_matrix)
{
  *rbf_matrix = rbf(particles, this->sigma);
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf_matrix->colwise().sum().transpose().array()) - (*rbf_matrix * particles).array();
  return dxkxy;
}

dynamic SBS::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector stddev = Eigen::VectorXd::Zero(particles.rows());

  Eigen::MatrixXd grads(particles.rows(), this->bounds.size());
  for (int j = 0; j < particles.rows(); j++)
  {
    double f_x;
    grads.row(j) = -this->k * gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  Eigen::MatrixXd kernel;
  Eigen::MatrixXd kernel_grad = this->rbf_grad(particles, &kernel);

  for (int i = 0; i < particles.rows(); i++)
  {
    double eval = f(particles.row(i));
    (*evals)[i] = eval;
  }
  Eigen::MatrixXd noise = zero_noise(particles.rows(), this->bounds.size());
  return {((kernel * grads + kernel_grad) / particles.rows()), stddev, noise};
}