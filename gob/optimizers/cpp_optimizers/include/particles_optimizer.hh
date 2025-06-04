/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizer.hh"
#include "schedulers.hh"
#include <memory>

#pragma once

struct dynamic
{
  Eigen::MatrixXd drift;
  dyn_vector stddev;
};

class Particles_Optimizer : public Optimizer
{
public:
  Particles_Optimizer(
      vec_bounds bounds,
      int n_particles = 200,
      int iter = 100,
      double dt = 0.01,
      int batch_size = 0,
      unique_ptr<Scheduler> sched = make_unique<Scheduler>()) : Optimizer(bounds, "Particles_Optimizer")
  {
    this->n_particles = n_particles;
    this->iter = iter;
    this->dt = dt;
    this->sched = move(sched);
    this->batch_size = batch_size;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

protected:
  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals) = 0;
  int n_particles;
  int iter;
  double dt;
  int batch_size;
  unique_ptr<Scheduler> sched;

private:
  void update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples);
};