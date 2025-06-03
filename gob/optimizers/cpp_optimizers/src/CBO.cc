/*
 * Created in 2025 by Gaëtan Serré
 */

#include "CBO.hh"
#include <iterator>

template <typename Iter>
typename std::iterator_traits<Iter>::value_type
log_sum_exp(Iter begin, Iter end)
{
  using VT = typename std::iterator_traits<Iter>::value_type;
  if (begin == end)
    return VT{};
  using std::exp;
  using std::log;
  auto max_elem = *std::max_element(begin, end);
  auto sum = std::accumulate(begin, end, VT{},
                             [max_elem](VT a, VT b)
                             { return a + exp(b - max_elem); });
  return max_elem + log(sum);
}

dyn_vector CBO::compute_vf(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{
  dyn_vector weights(particles.rows());
  for (int i = 0; i < particles.rows(); i++)
  {
    double f_x = f(particles.row(i));
    (*evals)[i] = f_x;
    weights[i] = -this->beta * f_x;
  }
  double lse = log_sum_exp(weights.data(), weights.data() + weights.size());

  dyn_vector vf = Eigen::VectorXd::Zero(particles.cols());
  for (int i = 0; i < particles.rows(); i++)
  {
    vf += exp(weights[i] - lse) * particles.row(i);
  }
  return vf;
}

double smooth_heaviside(double x)
{
  return 0.5 * erf(x) + 0.5;
}

Eigen::MatrixXd CBO::full_dynamics(const function<double(dyn_vector x)> &f, const int &time, const Eigen::MatrixXd &particles, vector<double> *evals)
{
  dyn_vector vf = compute_vf(particles, f, evals);
  double f_vf = f(vf);

  Eigen::MatrixXd dyn(particles.rows(), particles.cols());

  for (int i = 0; i < particles.rows(); i++)
  {
    dyn_vector diff = (particles.row(i) - vf.transpose());

    dyn_vector noise = diff.norm() * normal_random_vector(this->re, particles.row(i).cols(), 0, sqrt(this->dt));

    dyn.row(i) = -this->lambda * this->dt * diff * smooth_heaviside((1.0 / this->epsilon) * ((*evals)[i] - f_vf)) + this->sigma * noise;
  }

  return dyn;
}

Eigen::MatrixXd CBO::batch_dynamics(const function<double(dyn_vector x)> &f, const int &time, const Eigen::MatrixXd &particles, vector<double> *evals)
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
    dyn_vector vf = compute_vf(batch_particles, f, &batch_evals);

    for (int i = 0; i < M; i++)
    {
      (*evals)[perm[batch * M + i]] = batch_evals[i];
    }

    double f_vf = f(vf);

    for (int i = 0; i < M; i++)
    {
      dyn_vector diff = (batch_particles.row(i) - vf.transpose());

      dyn_vector noise = diff.norm() * normal_random_vector(this->re, batch_particles.row(i).cols(), 0, sqrt(this->dt));

      dyn.row(perm[batch * M + i]) = -this->lambda * this->dt * diff * smooth_heaviside((1.0 / this->epsilon) * (batch_evals[i] - f_vf)) + this->sigma * noise;
    }
  }

  return dyn;
}

Eigen::MatrixXd CBO::dynamics(const function<double(dyn_vector x)> &f, const int &time, const Eigen::MatrixXd &particles, vector<double> *evals)
{
  Eigen::MatrixXd dyn = this->use_batch ? this->batch_dynamics(f, time, particles, evals) : this->full_dynamics(f, time, particles, evals);

  this->beta = min(this->beta * 1.05, 100000.0);

  return dyn;
}