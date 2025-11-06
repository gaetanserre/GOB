/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Langevin.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

dynamic Langevin::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  Eigen::MatrixXd grads(particles.rows(), this->bounds.size());
  for (int j = 0; j < particles.rows(); j++)
  {
    double f_x;
    grads.row(j) = -gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  dyn_vector stddev = Eigen::VectorXd::Ones(particles.rows()) * sqrt(this->beta);
  Eigen::MatrixXd noise = normal_noise(particles.rows(), this->bounds.size(), this->re);
  return {grads, stddev, noise};
}