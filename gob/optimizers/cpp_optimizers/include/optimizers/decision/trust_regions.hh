/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"
#include "Cover_Tree.hh"

class Point
{
public:
  Point(dyn_vector x)
  {
    this->x = x;
  }

  double distance(const Point &p) const
  {
    return (this->x - p.x).norm();
  }

  double operator==(const Point &p) const
  {
    return this->x == p.x;
  }

  void print()
  {
    print_vector(this->x);
  }

  dyn_vector x;
};

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
      int max_trials,
      double region_radius,
      int bobyqa_eval,
      vector<void *> data,
      vector<void (*)(void)> functions,
      decision_f decision,
      callback_f callback = nullptr)
      : Optimizer(bounds, "Trust Regions")
  {
    this->n_eval = n_eval;
    this->max_trials = max_trials;
    this->region_radius = region_radius;
    this->bobyqa_eval = bobyqa_eval;
    this->data = data;
    this->functions = functions;
    this->decision = decision;
    this->callback = callback;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

private:
  int n_eval;
  int max_trials;
  double region_radius;
  int bobyqa_eval;
  vector<void *> data;
  vector<void (*)(void)> functions;
  decision_f decision;
  callback_f callback;
};