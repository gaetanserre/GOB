/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Langevin.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

dynamic Langevin::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  Eigen::MatrixXd grads(particles.rows(), particles.cols());
  for (int j = 0; j < particles.rows(); j++)
  {
    double f_x;
    grads.row(j) = -gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  Eigen::MatrixXd noise = normal_noise(particles.rows(), particles.cols(), this->re) * sqrt(this->beta);
  return {grads, noise};
}