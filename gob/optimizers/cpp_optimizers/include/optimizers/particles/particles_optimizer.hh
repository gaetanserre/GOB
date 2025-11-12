/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"
#include "optimizers/particles/schedulers.hh"

#pragma once

struct dynamic
{
  Eigen::MatrixXd drift;
  Eigen::MatrixXd noise;
};

class Particles_Optimizer : public Optimizer
{
public:
  Particles_Optimizer(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      int batch_size,
      Scheduler *sched = new Scheduler(),
      std::string name = "Particles Optimizer") : Optimizer(bounds, name)
  {
    this->n_particles = n_particles;
    this->iter = iter;
    this->dt = dt;
    this->sched = sched;
    this->batch_size = batch_size;
  }

  ~Particles_Optimizer()
  {
    delete sched;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals) = 0;
  int n_particles;
  int iter;
  double dt;
  int batch_size;
  Scheduler *sched;

private:
  void update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples);
};