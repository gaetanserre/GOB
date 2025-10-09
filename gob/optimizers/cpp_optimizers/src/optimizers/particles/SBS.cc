/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS.hh"
#include "optimizers/particles/noise.hh"

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

Eigen::MatrixXd pairwise_dist(const Eigen::MatrixXd &particles)
{
  // Create 0 square matrix
  Eigen::MatrixXd dists(particles.rows(), particles.rows());
  dists.setZero();
  for (int i = 0; i < particles.rows(); i++)
  {
    for (int j = i + 1; j < particles.rows(); j++)
    {
      double d = (particles.row(i) - particles.row(j)).norm();
      dists(i, j) = d;
      dists(j, i) = d;
    }
  }
  return dists;
}

Eigen::MatrixXd SBS::rbf(const Eigen::MatrixXd &particles)
{
  Eigen::MatrixXd pdists = pairwise_dist(particles);
  return (-pdists / (2 * this->sigma * this->sigma)).array().exp();
}

Eigen::MatrixXd SBS::rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf)
{
  *rbf = this->rbf(particles);
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf->colwise().sum().transpose().array()) - (*rbf * particles).array();
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