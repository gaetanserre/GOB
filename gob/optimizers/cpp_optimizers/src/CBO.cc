/*
 * Created in 2025 by Gaëtan Serré
 */

#include "CBO.hh"

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

dyn_vector CBO::compute_consensus(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector weights(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    double f_x = f(particles.row(i));
    (*evals)[i] = f_x;
    weights[i] = -this->beta * f_x;
  }
  double lse = log_sum_exp(weights.data(), weights.data() + weights.size());

  dyn_vector vf = Eigen::VectorXd::Zero(particles.cols());
  for (int i = 0; i < particles.rows(); i++)
  {
    vf += exp(weights[i] - lse) * particles.row(i);
  }
  return vf;
}

double smooth_heaviside(double x)
{
  return 0.5 * erf(x) + 0.5;
}

dynamic CBO::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector vf = compute_consensus(particles, f, evals);
  double f_vf = f(vf);

  Eigen::MatrixXd drift(particles.rows(), particles.cols());
  dyn_vector stddev(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    dyn_vector diff = (particles.row(i) - vf.transpose());

    stddev[i] = diff.norm() * this->sigma;
    drift.row(i) = -this->lambda * diff * smooth_heaviside((1.0 / this->epsilon) * ((*evals)[i] - f_vf));
  }
  this->beta = min(this->beta * 1.05, 100000.0);
  return {drift, stddev};
}