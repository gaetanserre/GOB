/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/common_noise.hh"
#include "optimizers/particles/common-noise/common_noise_utils.hh"

common_dynamic Common_Noise::m1_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  int d = particles.cols();
  return {dyn_vector::Zero(d), Eigen::MatrixXd::Ones(d, d)};
}

common_dynamic Common_Noise::square_dynamic(const Eigen::MatrixXd &particles, const int &idx, auto func)
{
  int d = particles.cols();
  dyn_vector drift = dyn_vector::Zero(d);
  dyn_vector noise = dyn_vector::Zero(d);
  for (int dim = 0; dim < d; dim++)
  {
    double moment = func(particles, dim);
    drift(dim) = (particles(idx, dim) * (moment - 3 / 2)) / (4 * pow(this->lambda + moment, 2));
    noise(dim) = particles(idx, dim) / (2 * (this->lambda + moment));
  }
  return {drift, noise.asDiagonal()};
}

common_dynamic Common_Noise::m2_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  auto moment_2 = [](const Eigen::MatrixXd &p, const int &dim)
  { return compute_moment(p, 2, dim); };
  return square_dynamic(particles, idx, moment_2);
}

common_dynamic Common_Noise::var_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  return square_dynamic(particles, idx, compute_variance);
}

void Common_Noise::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples)
{
  vector<double> evals((*particles).rows());
  dynamic dyn = this->base_opt->compute_dynamics(*particles, f, &evals);
  dyn_vector common_noise = normal_random_vector(this->base_opt->re, particles->cols(), 0, 1);

  for (int j = 0; j < particles->rows(); j++)
  {
    common_dynamic common_dynamic;
    if (this->noise_type == NoiseType::M1)
    {
      common_dynamic = this->m1_dynamic(*particles, j);
    }
    else if (this->noise_type == NoiseType::M2)
    {
      common_dynamic = this->m2_dynamic(*particles, j);
    }
    else if (this->noise_type == NoiseType::VAR)
    {
      common_dynamic = this->var_dynamic(*particles, j);
    }

    all_evals->push_back(evals[j]);
    samples->push_back((*particles).row(j));
    particles->row(j) += this->base_opt->dt *
                             (dyn.drift.row(j) + common_dynamic.drift.transpose()) +
                         sqrt(this->base_opt->dt) *
                             (dyn.noise.row(j) + this->gamma * (common_dynamic.noise * common_noise).transpose());
    particles->row(j) = clip_vector(particles->row(j), this->base_opt->bounds);
  }
}

result_eigen Common_Noise::minimize(function<double(dyn_vector)> f)
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