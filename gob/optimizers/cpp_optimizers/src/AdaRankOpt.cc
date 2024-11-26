/*
 * Created in 2024 by Gaëtan Serré
 */

#include "AdaRankOpt.hh"
#include "PolynomialFeatures.hh"
#include "Simplex.hh"

Eigen::MatrixXd AdaRankOpt::polynomial_matrix(vector<dyn_vector> &X, int degree)
{
  int n = X.size() - 1;
  int d = X[0].cols();
  int n_out = comp(d + degree, d) - 1;
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_out, n);
  for (int i = 0; i < n; i++)
  {
    dyn_vector poly = polynomial_features(X[i + 1], degree) - polynomial_features(X[i], degree);
    for (int j = 0; j < n_out; j++)
    {
      M(j, i) = poly(j);
    }
  }
  return M;
}

bool AdaRankOpt::is_polyhedral_set_empty(vector<dyn_vector> &X, int degree)
{
  Eigen::MatrixXd M = this->polynomial_matrix(X, degree);
  return simplex(M, this->param, this->simplex_tol);
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
      while (true)
      {
        dyn_vector x = unif_random_vector(this->re, this->bounds);
        double f_x_tmp = samples.back().second + 1;
        samples.push_back(make_pair(x, f_x_tmp));
        vector<dyn_vector> X(samples.size());
        for (int i = 0; i < samples.size(); i++)
        {
          X[i] = samples[i].first;
        }
        if (this->is_polyhedral_set_empty(X, degree))
        {
          samples[samples.size() - 1].second = -f(x);
          sort(samples.begin(), samples.end(), compare_pair);
          break;
        }
        else
          samples.pop_back();
      }
    }

    while (true)
    {
      vector<dyn_vector> X(samples.size());
      for (int i = 0; i < samples.size(); i++)
      {
        X[i] = samples[i].first;
      }
      if (this->is_polyhedral_set_empty(X, degree))
        break;

      degree++;
    }
  }

  return make_pair(samples.back().first, -samples.back().second);
}
