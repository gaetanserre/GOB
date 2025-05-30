/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizer.hh"

#pragma once

class Particles_Optimizer : public Optimizer
{
public:
  Particles_Optimizer(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      bool use_adam = true,
      double lr = 0.5) : Optimizer(bounds, "Particles_Optimizer")
  {
    this->n_particles = n_particles;
    this->iter = iter;
    this->use_adam = use_adam;
    this->lr = lr;
  };
  virtual result_eigen minimize(function<double(dyn_vector x)> f);
  virtual Eigen::MatrixXd dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals) = 0;

  int n_particles;
  int iter;
  bool use_adam;
  double lr;
};