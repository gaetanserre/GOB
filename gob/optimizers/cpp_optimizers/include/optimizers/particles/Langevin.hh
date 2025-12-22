/*
 * Created in 2025 by Gaëtan Serré
 */

#pragma once

#include "optimizers/particles/particles_optimizer.hh"

class Langevin : public Particles_Optimizer
{
public:
  Langevin(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, batch_size, new LinearScheduler(dt, 1), "Langevin")
  {
    this->beta = beta;
  }

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals, const int &time);

private:
  int k;
  double beta;
};