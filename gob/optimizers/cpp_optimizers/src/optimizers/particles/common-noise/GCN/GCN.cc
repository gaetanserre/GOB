/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/GCN/GCN.hh"
#include "optimizers/particles/particles_utils.hh"
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

Eigen::MatrixXd GCN::compute_noise(const Eigen::MatrixXd &particles)
{
  int d = particles.cols();
  Eigen::MatrixXd rbf_matrix = rbf(particles, this->sigma);

  // Compute the square root of the RBF matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(rbf_matrix);

  // Ensure positive semi-definiteness
  auto D = es.eigenvalues().cwiseMax(0);
  Eigen::MatrixXd K_sqrt = es.eigenvectors() * D.cwiseSqrt().asDiagonal() * es.eigenvectors().transpose();
  Eigen::MatrixXd K_sqrt_kron = Eigen::kroneckerProduct(K_sqrt, Eigen::MatrixXd::Identity(d, d));

  dyn_vector alphas_tmp = K_sqrt_kron * normal_random_vector(this->re, K_sqrt_kron.rows(), 0, 1);
  Eigen::MatrixXd alphas = Eigen::Map<Eigen::MatrixXd>(alphas_tmp.data(), d, particles.rows()).transpose();
  return alphas;
}

void GCN::update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples, const int &time_)
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

  Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(particles->rows(), particles->cols());
  // if (time_ <= this->base_opt->iter / 2)
  {
    noise = this->compute_noise(*particles);
  }

  // Independent noise
  if (this->independent_noise)
  {
    (*particles) += dyn.noise * sqrt(dt);
  }

  // Noise update
  for (int j = 0; j < particles->rows(); j++)
  {
    all_evals->push_back(evals[j]);
    particles->row(j) += sqrt(dt) * noise.row(j);
    particles->row(j) = clip_vector(particles->row(j), this->bounds);
  }
}

result_eigen GCN::minimize(function<double(dyn_vector)> f)
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