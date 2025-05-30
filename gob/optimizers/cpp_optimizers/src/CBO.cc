/*
 * Created in 2025 by Gaëtan Serré
 */

#include "CBO.hh"

dyn_vector compute_vf(Eigen::MatrixXd &particles, dyn_vector weights)
{
  dyn_vector v = Eigen::VectorXd::Zero(particles.cols());
  double sum_weights = 0;
  for (int i = 0; i < particles.rows(); i++)
  {
    v += particles.row(i) * weights(i);
    sum_weights += weights(i);
  }
  return v / sum_weights;
}

double smooth_heaviside(double x)
{
  return 0.5 * erf(x) + 0.5;
}

dyn_vector CBO::weights(Eigen::MatrixXd &particles, function<double(dyn_vector x)> f, vector<double> *evals)
{
  dyn_vector w(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    double f_x = f(particles.row(i));
    (*evals)[i] = f_x;
    w[i] = exp(-this->beta * f_x);
  }
  return w;
}

Eigen::MatrixXd CBO::full_dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals)
{
  dyn_vector weights = this->weights(particles, f, evals);
  dyn_vector vf = compute_vf(particles, weights);
  if (contains_nan(vf))
  {
    throw runtime_error("CBO: Weights are all 0s. Consider decreasing beta.");
  }
  double f_vf = f(vf);

  Eigen::MatrixXd dyn(particles.rows(), particles.cols());

  for (int i = 0; i < particles.rows(); i++)
  {
    dyn_vector diff = (particles.row(i) - vf.transpose());

    dyn_vector noise(particles.row(i).cols());
    for (int j = 0; j < particles.row(i).cols(); j++)
    {
      noise[j] = diff[j] * unif_random_normal(this->re, 0, this->lambda);
    }

    dyn.row(i) = -this->lambda * diff * smooth_heaviside((1.0 / this->epsilon) * ((*evals)[i] - f_vf)) + sqrt(2) * this->sigma * noise;
  }

  return dyn;
}

Eigen::MatrixXd CBO::batch_dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals)
{
  vector<int> perm(particles.rows());
  for (size_t i = 0; i < perm.size(); ++i)
  {
    perm[i] = i;
  }
  std::shuffle(perm.begin(), perm.end(), this->re);

  Eigen::MatrixXd dyn(particles.rows(), particles.cols());
  int M = 5;
  for (int batch = 0; batch < particles.rows() / M; batch++)
  {
    Eigen::MatrixXd batch_particles(M, particles.cols());
    for (int i = 0; i < M; i++)
    {
      batch_particles.row(i) = particles.row(perm[batch * M + i]);
    }
    vector<double> batch_evals(M);
    dyn_vector weights = this->weights(batch_particles, f, &batch_evals);

    for (int i = 0; i < M; i++)
    {
      (*evals)[perm[batch * M + i]] = batch_evals[i];
    }
    dyn_vector vf = compute_vf(batch_particles, weights);

    if (contains_nan(vf))
    {
      throw runtime_error("CBO: Weights are all 0s. Consider decreasing beta.");
    }
    double f_vf = f(vf);

    for (int i = 0; i < M; i++)
    {
      dyn_vector diff = (batch_particles.row(i) - vf.transpose());

      dyn_vector noise(batch_particles.row(i).cols());
      for (int j = 0; j < batch_particles.row(i).cols(); j++)
      {
        noise[j] = diff[j] * unif_random_normal(this->re, 0, this->lambda);
      }

      dyn.row(perm[batch * M + i]) = -this->lambda * diff * smooth_heaviside((1.0 / this->epsilon) * (batch_evals[i] - f_vf)) + sqrt(2) * this->sigma * noise;
    }
  }

  return dyn;
}

Eigen::MatrixXd CBO::dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals)
{
  if (this->use_batch)
  {
    return this->batch_dynamics(f, time, particles, evals);
  }
  else
  {
    return this->full_dynamics(f, time, particles, evals);
  }
}