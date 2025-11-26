/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Full_Noise.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

dynamic Full_Noise::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  Eigen::MatrixXd drift = Eigen::MatrixXd::Ones(particles.rows(), particles.cols());
  Eigen::MatrixXd noise = normal_noise(particles.rows(), this->bounds.size(), this->re);
  return {drift, noise};
}