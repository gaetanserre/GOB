/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/noise.hh"

Eigen::MatrixXd zero_noise(const int &rows, const int &cols)
{
  return Eigen::MatrixXd::Zero(rows, cols);
}

Eigen::MatrixXd normal_noise(const int &rows, const int &cols, std::mt19937_64 re, double mean, double stddev)
{
  Eigen::MatrixXd noise(rows, cols);
  for (int i = 0; i < rows; i++)
  {
    dyn_vector n = normal_random_vector(re, cols, mean, stddev);
    noise.row(i) = n;
  }
  return noise;
}