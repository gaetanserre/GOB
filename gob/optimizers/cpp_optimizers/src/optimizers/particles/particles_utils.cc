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

Eigen::MatrixXd pairwise_dist(const Eigen::MatrixXd &particles)
{
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

Eigen::MatrixXd rbf(const Eigen::MatrixXd &particles, const double &sigma)
{
  Eigen::MatrixXd pdists = pairwise_dist(particles);
  return (-pdists / (2 * sigma * sigma)).array().exp();
}

dyn_vector gradient(dyn_vector x, const function<double(dyn_vector x)> &f, double *f_x, double tol)
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