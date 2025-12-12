/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

class Full_Noise : public Particles_Optimizer
{
public:
  Full_Noise(
      vec_bounds bounds,
      int n_particles,
      int iter,
      double dt,
      double alpha,
      int batch_size) : Particles_Optimizer(bounds, n_particles, iter, batch_size, new LinearScheduler(dt, alpha), "Full-Noise") {}

  virtual dynamic compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals, const int &time);
};