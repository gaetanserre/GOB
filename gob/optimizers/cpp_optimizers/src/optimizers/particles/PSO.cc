/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/PSO.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"

dynamic PSO::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector vf;
  if (this->beta > 0)
    vf = compute_consensus(particles, f, evals, this->beta);
  else
  {
    double min_eval = INFINITY;
    int argmin = -1;
    for (int i = 0; i < particles.rows(); i++)
    {
      double f_x = f(particles.row(i));
      (*evals)[i] = f_x;
      if (f_x < min_eval)
      {
        min_eval = f_x;
        argmin = i;
      }
    }
    vf = particles.row(argmin);
  }

  int i = 0;

  for (int i = 0; i < particles.rows(); i++)
  {
    this->velocities.row(i) = this->omega * this->velocities.row(i) + this->c2 * (vf.transpose() - particles.row(i));
  }

  Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(particles.rows(), this->bounds.size());
  return {this->velocities, noise};
}