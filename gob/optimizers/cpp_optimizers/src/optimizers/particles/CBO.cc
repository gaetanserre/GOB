/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/CBO.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

double smooth_heaviside(double x)
{
  return 0.5 * erf(x) + 0.5;
}

dynamic CBO::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector vf = compute_consensus(particles, f, evals, this->beta);
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
  Eigen::MatrixXd noise = normal_noise(particles.rows(), this->bounds.size(), this->re);
  return {drift, stddev, noise};
}