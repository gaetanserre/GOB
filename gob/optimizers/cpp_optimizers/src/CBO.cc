/*
 * Created in 2025 by Gaëtan Serré
 */

#include "CBO.hh"

dyn_vector compute_vf(Eigen::MatrixXd &particles, dyn_vector weights)
{
  dyn_vector v = Eigen::VectorXd::Zero(particles.cols());
  double sum_weights = 0;
  for (int i = 0; i < particles.rows(); i++)
  {
    v += particles.row(i) * weights(i);
    sum_weights += weights(i);
  }
  return v / sum_weights;
}

double smooth_heaviside(double x)
{
  return (1 / 2) * erf(x) + 1 / 2;
}

dyn_vector CBO::weights(Eigen::MatrixXd &particles, function<double(dyn_vector x)> f, vector<double> *evals)
{
  dyn_vector w(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    double f_x = f(particles.row(i));
    (*evals)[i] = f_x;
    w[i] = exp(-this->alpha * f_x);
  }
  return w;
}

Eigen::MatrixXd CBO::dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals)
{
  dyn_vector vf = compute_vf(particles, this->weights(particles, f, evals));
  double f_vf = f(vf);

  Eigen::MatrixXd dyn(particles.rows(), particles.cols());

  for (int i = 0; i < particles.rows(); i++)
  {
    dyn_vector diff = vf.transpose() - particles.row(i);
    dyn.row(i) = -this->lambda * diff * smooth_heaviside((1.0 / this->epsilon) * ((*evals)[i] - f_vf)) + sqrt(2) * this->sigma * diff * unif_random_normal(this->re, 0, this->lambda);
  }

  return dyn;
}
