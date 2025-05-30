/*
 * Created in 2025 by Gaëtan Serré
 */

#include "particles_optimizer.hh"
#include "Adam.hh"

result_eigen Particles_Optimizer::minimize(function<double(dyn_vector x)> f)
{
  Adam optimizer(this->n_particles, this->bounds.size(), this->lr);
  vector<double> all_evals;
  vector<dyn_vector> samples;
  Eigen::MatrixXd particles(this->n_particles, this->bounds.size());
  for (int i = 0; i < this->n_particles; i++)
  {
    particles.row(i) = unif_random_vector(this->re, this->bounds);
  }
  for (int i = 0; i < this->iter; i++)
  {
    vector<double> evals(this->n_particles);
    Eigen::MatrixXd dyns = this->dynamics(f, i, particles, &evals);
    for (int j = 0; j < this->n_particles; j++)
    {
      all_evals.push_back(evals[j]);
      samples.push_back(particles.row(j));
    }
    if (this->use_adam)
      particles = optimizer.step(dyns, particles);
    else
      particles += dyns;

    for (int j = 0; j < this->n_particles; j++)
      particles.row(j) = clip_vector(particles.row(j), this->bounds);

    if (this->has_stop_criterion && min_vec(all_evals) <= this->stop_criterion)
      break;
  }
  int argmin = argmin_vec(all_evals);
  return {samples[argmin], all_evals[argmin]};
}