/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/decision/ECP.hh"
#include "optimizers/decision/trust_regions.hh"

namespace ECP_trust
{
  bool decision(
      vector<pair<dyn_vector, double>> samples,
      dyn_vector x, vector<void *> data,
      vector<void (*)(void)> functions)
  {
    if (samples.size() == 0)
      return true;

    double *epsilon = (double *)data[0];
    int *h1 = (int *)data[1];
    int *h2 = (int *)data[2];
    int *C = (int *)data[3];
    double *theta = (double *)data[4];

    *h2 = *h2 + 1;
    if (*h2 - *h1 > *C)
    {
      *epsilon = *epsilon * *theta;
      *h2 = 0;
    }

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
      norms[i] = values[i] + *epsilon * (x - points[i]).norm();
    }
    bool res = max_values <= min_vec(norms);
    if (res)
    {
      *h1 = *h2;
      *epsilon = *epsilon * *theta;
      *h2 = 0;
    }
    return res;
  }
};

result_eigen ECP::minimize(function<double(dyn_vector x)> f)
{
  int h1 = 1, h2 = 0;

  vector<void *> data(5);
  data[0] = (void *)&this->epsilon;
  data[1] = (void *)&h1;
  data[2] = (void *)&h2;
  data[3] = (void *)&this->C;
  data[4] = (void *)&this->theta;

  vector<void (*)(void)> functions(0);

  TrustRegions tr = TrustRegions(
      this->bounds,
      this->n_eval,
      10000000,
      this->trust_region_radius,
      this->bobyqa_eval,
      data,
      functions,
      &ECP_trust::decision);

  if (this->has_stop_criterion)
    tr.set_stop_criterion(this->stop_criterion);

  return tr.minimize(f);
}