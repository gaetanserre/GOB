/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/decision/AdaLIPO_P.hh"
#include "optimizers/decision/trust_regions.hh"

namespace AdaLIPO_P_trust
{
  bool decision(
      vector<pair<dyn_vector, double>> samples,
      dyn_vector x, vector<void *> data,
      vector<void (*)(void)> functions)
  {
    if (samples.size() == 0)
      return true;
    double *k_hat = (double *)data[0];
    vector<dyn_vector> points(samples.size());
    vector<double> values(samples.size());
    for (int i = 0; i < samples.size(); i++)
    {
      points[i] = samples[i].first;
      values[i] = samples[i].second;
    }

    double max_values = max_vec(values);
    vector<double> norms(points.size());
    for (int i = 0; i < points.size(); i++)
    {
      norms[i] = values[i] + *k_hat * (x - points[i]).norm();
    }
    return max_values <= min_vec(norms);
  }

  void callback(
      vector<pair<dyn_vector, double>> samples,
      vector<void *> data,
      vector<void (*)(void)> functions)
  {
    if (samples.size() >= 2)
    {
      double *k_hat = (double *)data[0];
      vector<double> *ratios = (vector<double> *)data[1];
      vector<dyn_vector> points(samples.size());
      vector<double> values(samples.size());
      for (int i = 0; i < samples.size(); i++)
      {
        points[i] = samples[i].first;
        values[i] = samples[i].second;
      }

      double alpha = 1e-2;
      dyn_vector x = points.back();
      double value = values.back();
      for (int i = 0; i < samples.size() - 1; i++)
      {
        (*ratios).push_back(abs(value - values[i]) / (x - points[i]).norm());
      }
      int i = ceil(log(max_vec(*ratios)) / log(1 + alpha));
      *k_hat = pow(1 + alpha, i);
    }
  }
}

result_eigen AdaLIPO_P::minimize(function<double(dyn_vector x)> f)
{
  double k_hat = 0;
  vector<double> ratios;

  vector<void *> data(2);
  data[0] = (void *)&k_hat;
  data[1] = (void *)&ratios;

  vector<void (*)(void)> functions(0);

  TrustRegions tr = TrustRegions(
      this->bounds,
      this->n_eval,
      this->max_trials,
      this->trust_region_radius,
      this->bobyqa_eval,
      data,
      functions,
      &AdaLIPO_P_trust::decision,
      &AdaLIPO_P_trust::callback);

  if (this->has_stop_criterion)
    tr.set_stop_criterion(this->stop_criterion);

  return tr.minimize(f);
}