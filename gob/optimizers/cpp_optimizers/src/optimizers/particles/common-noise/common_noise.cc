/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/common_noise.hh"

dynamic Common_Noise::m1_dynamic(const Eigen::MatrixXd &particles)
{
  int d = particles.cols();
  dyn_vector noise = normal_random_vector(this->re, d, 0, 1);
  Eigen::MatrixXd common_noise = Eigen::MatrixXd::Zero(particles.rows(), d);
  for (int i = 0; i < particles.rows(); i++)
  {
    common_noise.row(i) = noise.transpose();
  }
  return {Eigen::MatrixXd::Zero(particles.rows(), d), common_noise};
}

dynamic Common_Noise::m2_dynamic(const Eigen::MatrixXd &particles)
{
  int d = particles.cols();
  dyn_vector noise = normal_random_vector(this->re, d, 0, 1);
  Eigen::MatrixXd common_noise = Eigen::MatrixXd::Zero(particles.rows(), d);
  return {Eigen::MatrixXd::Zero(particles.rows(), d), common_noise};
}

void Common_Noise::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples)
{
  vector<double> evals((*particles).rows());
  dynamic dyn = this->base_opt->compute_dynamics(*particles, f, &evals);

  dynamic common_dynamic;
  if (this->noise_type == NoiseType::M1)
  {
    common_dynamic = this->m1_dynamic(*particles);
  }
  else if (this->noise_type == NoiseType::M2)
  {
    common_dynamic = this->m2_dynamic(*particles);
  }

  for (int j = 0; j < (*particles).rows(); j++)
  {
    all_evals->push_back(evals[j]);
    samples->push_back((*particles).row(j));
    particles->row(j) += (dyn.drift.row(j) + common_dynamic.drift.row(j)) * this->base_opt->dt + sqrt(this->base_opt->dt) * (dyn.noise.row(j) + this->gamma * common_dynamic.noise.row(j));
    particles->row(j) = clip_vector(particles->row(j), this->base_opt->bounds);
  }
}

result_eigen Common_Noise::minimize(function<double(dyn_vector x)> f)
{
  vector<double> all_evals;
  vector<dyn_vector> samples;
  Eigen::MatrixXd particles(this->base_opt->n_particles, this->base_opt->bounds.size());
  for (int i = 0; i < this->base_opt->n_particles; i++)
  {
    particles.row(i) = unif_random_vector(this->base_opt->re, this->base_opt->bounds);
  }
  for (int i = 0; i < this->base_opt->iter; i++)
  {
    this->update_particles(&particles, f, &all_evals, &samples);

    if (this->base_opt->has_stop_criterion && min_vec(all_evals) <= this->base_opt->stop_criterion)
      break;
    this->base_opt->sched->step();
  }
  int argmin = argmin_vec(all_evals);
  this->base_opt->sched->reset();
  return {samples[argmin], all_evals[argmin]};
}