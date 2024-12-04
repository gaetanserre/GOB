/*
 * Created in 2024 by Gaëtan Serré
 */

#include "AdaRankOpt.hh"
#include "PolynomialFeatures.hh"
#include "Simplex.hh"

Eigen::MatrixXd AdaRankOpt::polynomial_matrix(vector<pair<dyn_vector, double>> &samples, int degree)
{
  int n = samples.size() - 1;
  int d = samples[0].first.size();
  int n_out = comp(d + degree, d) - 1;
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_out, n);
  for (int i = 0; i < n; i++)
  {
    dyn_vector poly = polynomial_features(samples[i + 1].first, degree) - polynomial_features(samples[i].first, degree);
    for (int j = 0; j < n_out; j++)
    {
      M(j, i) = poly(j);
    }
  }
  return M;
}

bool AdaRankOpt::is_polyhedral_set_empty(vector<pair<dyn_vector, double>> &samples, int degree)
{
  Eigen::MatrixXd M = this->polynomial_matrix(samples, degree);
  return simplex(M, this->param) == GLP_NOFEAS;
}

result_eigen AdaRankOpt::minimize(function<double(dyn_vector x)> f)
{
  int degree = 1;

  vector<pair<dyn_vector, double>> samples;
  dyn_vector x = unif_random_vector(this->re, this->bounds);
  samples.push_back(make_pair(x, -f(x)));

  auto compare_pair = [](pair<dyn_vector, double> a, pair<dyn_vector, double> b) -> bool
  {
    return a.second < b.second;
  };

  sort(samples.begin(), samples.end(), compare_pair);

  for (int t = 1; t < this->n_eval; t++)
  {
    if (Bernoulli(this->re, 0.1))
    {
      dyn_vector x = unif_random_vector(this->re, this->bounds);
      samples.push_back(make_pair(x, -f(x)));
      sort(samples.begin(), samples.end(), compare_pair);
    }
    else
    {
      int nb_samples = 0;
      while (true)
      {
        dyn_vector x = unif_random_vector(this->re, this->bounds);
        double f_x_tmp = samples.back().second + 1;
        samples.push_back(make_pair(x, f_x_tmp));
        if (this->is_polyhedral_set_empty(samples, degree))
        {
          samples[samples.size() - 1].second = -f(x);
          sort(samples.begin(), samples.end(), compare_pair);
          break;
        }
        else
          samples.pop_back();

        nb_samples++;

        if (nb_samples > this->max_samples)
        {
          if (this->verbose)
            printf("Warning: AdaRankOpt could not converge. Early stopping at iteration %d with degree %d.\n", t, degree);
          return make_pair(samples.back().first, -samples.back().second);
        }
      }
    }

    while (degree < this->max_degree)
    {
      if (this->is_polyhedral_set_empty(samples, degree))
        break;

      degree++;
    }
  }
  return make_pair(samples.back().first, -samples.back().second);
}
