/*
 * Created in 2025 by Gaëtan Serré
 */

#include "particles_optimizer.hh"

void Particles_Optimizer::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples)
{
  vector<double> evals((*particles).rows());
  dynamic dyn = this->compute_dynamics(*particles, f, &evals);
  for (int j = 0; j < (*particles).rows(); j++)
  {
    all_evals->push_back(evals[j]);
    samples->push_back((*particles).row(j));
    dyn_vector noise = normal_random_vector(this->re, particles->row(j).cols(), 0, 1);
    particles->row(j) += dyn.drift.row(j) * this->dt + dyn.stddev[j] * sqrt(this->dt) * noise.transpose();
    particles->row(j) = clip_vector(particles->row(j), this->bounds);
  }
}

result_eigen Particles_Optimizer::minimize(function<double(dyn_vector x)> f)
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
        throw runtime_error(format("Batch size ({}) cannot be larger than the number of particles ({}).", this->batch_size, this->n_particles));
      }
      if (this->n_particles % this->batch_size != 0)
      {
        throw runtime_error(format("Number of particles ({}) must be a multiple of the batch size ({}).", this->n_particles, this->batch_size));
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
      this->update_particles(&batch_particles, f, &all_evals, &samples);
      for (int j = 0; j < this->batch_size; j++)
      {
        particles.row(perm[j]) = batch_particles.row(j);
      }
    }
    else

      this->update_particles(&particles, f, &all_evals, &samples);

    if (this->has_stop_criterion && min_vec(all_evals) <= this->stop_criterion)
      break;
    this->sched->step();
  }
  int argmin = argmin_vec(all_evals);
  return {samples[argmin], all_evals[argmin]};
}