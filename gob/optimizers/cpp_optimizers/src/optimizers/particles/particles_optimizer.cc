/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

void Particles_Optimizer::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples, int &t)
{
  vector<double> evals((*particles).rows());
  for (int i = 0; i < particles->rows(); i++)
  {
    samples->push_back((*particles).row(i));
  }

  dynamic dyn = this->compute_dynamics(*particles, f, &evals, t);

  // Drift update
  this->sched->step(particles, dyn.drift, t);
  double dt = this->sched->get_dt();

  // Noise update
  for (int j = 0; j < particles->rows(); j++)
  {
    all_evals->push_back(evals[j]);
    particles->row(j) += sqrt(dt) * dyn.noise.row(j);
    particles->row(j) = clip_vector(particles->row(j), this->bounds);
  }
}

result_eigen Particles_Optimizer::minimize(function<double(dyn_vector)> f)
{
  vector<double> all_evals;
  vector<dyn_vector> samples;
  Eigen::MatrixXd particles(this->n_particles, this->bounds.size());

  for (int i = 0; i < this->n_particles; i++)
  {
    particles.row(i) = unif_random_vector(this->re, this->bounds);
  }
  for (int i = 0; i < this->iter; i++)
  {
    if (this->batch_size > 0)
    {
      if (this->n_particles < this->batch_size)
      {
        string msg = "Batch size (" + to_string(this->batch_size) + ") cannot be larger than the number of particles (" + to_string(this->n_particles) + ").";
        throw runtime_error(msg);
      }
      if (this->n_particles % this->batch_size != 0)
      {
        string msg = "Number of particles (" + to_string(this->n_particles) + ") must be a multiple of the batch size (" + to_string(this->batch_size) + ").";
        throw runtime_error(msg);
      }

      vector<int> perm(particles.rows());
      for (size_t j = 0; j < perm.size(); ++j)
      {
        perm[j] = j;
      }
      std::shuffle(perm.begin(), perm.end(), this->re);

      Eigen::MatrixXd batch_particles(this->batch_size, particles.cols());
      for (int j = 0; j < this->batch_size; j++)
      {
        batch_particles.row(j) = particles.row(perm[j]);
      }
      this->update_particles(&batch_particles, f, &all_evals, &samples, i);
      for (int j = 0; j < this->batch_size; j++)
      {
        particles.row(perm[j]) = batch_particles.row(j);
      }
    }
    else

      this->update_particles(&particles, f, &all_evals, &samples, i);

    if (this->has_stop_criterion && min_vec(all_evals) <= this->stop_criterion)
      break;
  }
  int argmin = argmin_vec(all_evals);
  return {samples[argmin], all_evals[argmin]};
}