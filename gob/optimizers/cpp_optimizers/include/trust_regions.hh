/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizer.hh"

typedef bool (*decision_f)(
    vector<pair<dyn_vector, double>>,
    dyn_vector, vector<void *>,
    vector<void (*)(void)>);

typedef void (*callback_f)(
    vector<pair<dyn_vector, double>>,
    vector<void *>,
    vector<void (*)(void)>);

class TrustRegions : public Optimizer
{
public:
  TrustRegions(
      vec_bounds bounds,
      int n_eval,
      int max_samples,
      double region_radius,
      int bobyqa_eval,
      vector<void *> data,
      vector<void (*)(void)> functions,
      decision_f decision,
      callback_f callback = nullptr)
      : Optimizer(bounds, "Trust Regions")
  {
    this->n_eval = n_eval;
    this->max_samples = max_samples;
    this->region_radius = region_radius;
    this->bobyqa_eval = bobyqa_eval;
    this->data = data;
    this->functions = functions;
    this->decision = decision;
    this->callback = callback;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  int max_samples;
  double region_radius;
  int bobyqa_eval;
  vector<void *> data;
  vector<void (*)(void)> functions;
  decision_f decision;
  callback_f callback;
};