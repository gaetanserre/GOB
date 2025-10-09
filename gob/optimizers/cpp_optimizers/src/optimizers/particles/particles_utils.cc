/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_utils.hh"

double log_sum_exp(double *begin, double *end)
{
  if (begin == end)
    return 0;
  double max_elem = *max_element(begin, end);
  double sum = accumulate(begin, end, 0,
                          [max_elem](double a, double b)
                          { return a + exp(b - max_elem); });
  return max_elem + log(sum);
}

dyn_vector compute_consensus(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals, double &beta)
{
  dyn_vector weights(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    double f_x = f(particles.row(i));
    (*evals)[i] = f_x;
    weights[i] = -beta * f_x;
  }
  double lse = log_sum_exp(weights.data(), weights.data() + weights.size());

  dyn_vector vf = Eigen::VectorXd::Zero(particles.cols());
  for (int i = 0; i < particles.rows(); i++)
  {
    vf += exp(weights[i] - lse) * particles.row(i);
  }
  return vf;
}