/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/common_noise_utils.hh"

double compute_moment(const Eigen::MatrixXd &particles, const int &r, const int &dim)
{
  double moment = 0.0;
  for (int i = 0; i < particles.rows(); i++)
  {
    moment += pow(particles(i, dim), r);
  }
  moment /= particles.rows();
  return moment;
}

double compute_variance(const Eigen::MatrixXd &particles, const int &dim)
{
  double mean = compute_moment(particles, 1, dim);
  double variance = 0.0;
  for (int i = 0; i < particles.rows(); i++)
  {
    variance += pow(particles(i, dim) - mean, 2);
  }
  variance /= particles.rows();
  return variance;
}