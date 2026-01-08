/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/SMD/SMD.hh"
#include "optimizers/particles/common-noise/SMD/SMD_utils.hh"

common_dynamic SMD::m1_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  int d = particles.cols();
  return {dyn_vector::Zero(d), (1 / (1 + this->lambda)) * Eigen::MatrixXd::Identity(d, d)};
}

common_dynamic SMD::m2_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  int d = particles.cols();
  dyn_vector drift = dyn_vector::Zero(d);
  dyn_vector noise = dyn_vector::Zero(d);
  for (int dim = 0; dim < d; dim++)
  {
    double moment = compute_moment(particles, 2, dim);
    drift(dim) = particles(idx, dim) * (this->delta - 3 / 2) / (4 * pow(this->lambda + moment, 2));
    noise(dim) = particles(idx, dim) / (2 * (this->lambda + moment));
  }
  return {drift, noise.asDiagonal()};
}

common_dynamic SMD::var_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  int d = particles.cols();
  dyn_vector drift = dyn_vector::Zero(d);
  dyn_vector noise = dyn_vector::Zero(d);
  for (int dim = 0; dim < d; dim++)
  {
    double res[2];
    compute_mean_variance(particles, dim, res);
    drift(dim) = (particles(idx, dim) - res[0]) * (this->delta - 3 / 2) / (4 * pow(this->lambda + res[1], 2));
    noise(dim) = (particles(idx, dim) - res[0]) / (2 * (this->lambda + res[1]));
  }
  return {drift, noise.asDiagonal()};
}

common_dynamic SMD::mean_var_dynamic(const Eigen::MatrixXd &particles, const int &idx)
{
  int d = particles.cols();
  common_dynamic var_dyn = this->var_dynamic(particles, idx);
  Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(d, 2 * d);
  noise << Eigen::MatrixXd::Identity(d, d), var_dyn.noise;
  return {var_dyn.drift, noise};
}

int get_common_dim(NoiseType noise_type, int d)
{
  if (noise_type == NoiseType::MVAR)
  {
    return 2 * d;
  }
  else
  {
    return d;
  }
}

void SMD::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples, const int &time_)
{
  vector<double> evals((*particles).rows());
  for (int i = 0; i < particles->rows(); i++)
  {
    samples->push_back((*particles).row(i));
  }

  dynamic dyn = this->base_opt->compute_dynamics(*particles, f, &evals, time_);

  // Drift update
  this->base_opt->sched->step(particles, dyn.drift, time_);
  double dt = this->base_opt->sched->get_dt();

  // Noise update
  if (this->independent_noise)
  {
    (*particles) += dyn.noise * sqrt(dt);
  }

  dyn_vector common_noise = normal_random_vector(
      this->base_opt->re,
      get_common_dim(this->noise_type, particles->cols()),
      0, 1);

  for (int j = 0; j < particles->rows(); j++)
  {
    all_evals->push_back(evals[j]);
    samples->push_back((*particles).row(j));

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
    else if (this->noise_type == NoiseType::MVAR)
    {
      common_dynamic = this->mean_var_dynamic(*particles, j);
    }

    // Noise, common drift, and common noise update
    particles->row(j) += dt * this->gamma * common_dynamic.drift.transpose() + sqrt(dt) * this->gamma * (common_dynamic.noise * common_noise).transpose();

    particles->row(j) = clip_vector(particles->row(j), this->base_opt->bounds);
  }
}

result_eigen SMD::minimize(function<double(dyn_vector)> f)
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
    this->update_particles(&particles, f, &all_evals, &samples, i);

    if (this->base_opt->has_stop_criterion && min_vec(all_evals) <= this->base_opt->stop_criterion)
      break;
  }
  int argmin = argmin_vec(all_evals);
  return {samples[argmin], all_evals[argmin]};
}