/*
 * Created in 2025 by Gaëtan Serré
 */

#include "SBS.hh"

dyn_vector gradient(dyn_vector x, function<double(dyn_vector x)> &f, double *f_x, double tol = 1e-9)
{
  dyn_vector grad(x.size());
  *f_x = f(x);
  for (int i = 0; i < x.size(); i++)
  {
    dyn_vector x_plus = x;
    x_plus[i] += tol;
    grad(i) = ((f(x_plus) - *f_x) / tol);
  }
  return grad;
}

Eigen::MatrixXd pairwise_dist(Eigen::MatrixXd &particles)
{
  // Create 0 square matrix
  Eigen::MatrixXd dists(particles.rows(), particles.rows());
  dists.setZero();
  for (int i = 0; i < particles.rows(); i++)
  {
    for (int j = i + 1; j < particles.rows(); j++)
    {
      double d = (particles.row(i) - particles.row(j)).norm();
      dists(i, j) = d;
      dists(j, i) = d;
    }
  }
  return dists;
}

Eigen::MatrixXd SBS::rbf(Eigen::MatrixXd &particles)
{
  Eigen::MatrixXd pdists = pairwise_dist(particles);
  return (-pdists / (2 * this->sigma * this->sigma)).array().exp();
}

Eigen::MatrixXd SBS::rbf_grad(Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf)
{
  *rbf = this->rbf(particles);
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf->colwise().sum().transpose().array()) - (*rbf * particles).array();
  return dxkxy;
}

Eigen::MatrixXd SBS::dynamics(function<double(dyn_vector x)> f, int &time, Eigen::MatrixXd &particles, vector<double> *evals)
{
  Eigen::MatrixXd grads(this->n_particles, this->bounds.size());
  for (int j = 0; j < this->n_particles; j++)
  {
    double f_x;
    grads.row(j) = -this->k * gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  Eigen::MatrixXd kernel;
  Eigen::MatrixXd kernel_grad = this->rbf_grad(particles, &kernel);

  for (int i = 0; i < particles.rows(); i++)
  {
    double eval = f(particles.row(i));
    (*evals)[i] = eval;
  }
  return -((kernel * grads + kernel_grad) / this->n_particles);
}

/* #include "SBS.hh"
#include "Adam.hh"

dyn_vector gradient(dyn_vector x, function<double(dyn_vector x)> &f, double *f_x, double tol = 1e-9)
{
  dyn_vector grad(x.size());
  *f_x = f(x);
  for (int i = 0; i < x.size(); i++)
  {
    dyn_vector x_plus = x;
    x_plus[i] += tol;
    grad(i) = ((f(x_plus) - *f_x) / tol);
  }
  return grad;
}

Eigen::MatrixXd pairwise_dist(Eigen::MatrixXd &X)
{
  // Create 0 square matrix
  Eigen::MatrixXd dists(X.rows(), X.rows());
  dists.setZero();
  for (int i = 0; i < X.rows(); i++)
  {
    for (int j = i + 1; j < X.rows(); j++)
    {
      double d = (X.row(i) - X.row(j)).norm();
      dists(i, j) = d;
      dists(j, i) = d;
    }
  }
  return dists;
}

Eigen::MatrixXd SBS::rbf(Eigen::MatrixXd &X)
{
  Eigen::MatrixXd pdists = pairwise_dist(X);
  return (-pdists / (2 * this->sigma * this->sigma)).array().exp();
}

Eigen::MatrixXd SBS::rbf_grad(Eigen::MatrixXd &X, Eigen::MatrixXd *rbf)
{
  *rbf = this->rbf(X);
  Eigen::MatrixXd dxkxy = (X.array().colwise() * rbf->colwise().sum().transpose().array()) - (*rbf * X).array();
  return dxkxy;
}

result_eigen SBS::minimize(function<double(dyn_vector x)> f)
{
  Adam optimizer(this->n_particles, this->bounds.size(), this->lr);
  vector<double> all_evals;
  vector<dyn_vector> samples;
  Eigen::MatrixXd particles(this->n_particles, this->bounds.size());
  for (int i = 0; i < this->n_particles; i++)
  {
    particles.row(i) = unif_random_vector(this->re, this->bounds);
  }
  for (double k : this->k_iter)
  {
    for (int i = 0; i < this->svgd_iter; i++)
    {
      Eigen::MatrixXd grads(this->n_particles, this->bounds.size());
      for (int j = 0; j < this->n_particles; j++)
      {
        double f_x;
        grads.row(j) = -k * gradient(particles.row(j), f, &f_x);
        all_evals.push_back(f_x);
        samples.push_back(particles.row(j));
      }
      Eigen::MatrixXd kernel;
      Eigen::MatrixXd kernel_grad = this->rbf_grad(particles, &kernel);
      particles = optimizer.step(-((kernel * grads + kernel_grad) / this->n_particles), particles);

      for (int j = 0; j < this->n_particles; j++)
        particles.row(j) = clip_vector(particles.row(j), this->bounds);

      if (this->has_stop_criterion && min_vec(all_evals) <= this->stop_criterion)
        break;
    }
  }
  int argmin = argmin_vec(all_evals);
  return {samples[argmin], all_evals[argmin]};
} */