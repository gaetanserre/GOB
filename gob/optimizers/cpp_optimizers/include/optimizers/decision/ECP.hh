/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"
#include <deque>

class ECP : public Optimizer
{
public:
  ECP(vec_bounds bounds,
      int n_eval,
      double epsilon,
      double theta_init,
      int C,
      int max_trials,
      double trust_region_radius,
      int bobyqa_eval,
      bool verbose = false) : Optimizer(bounds, "ECP+TR")
  {
    this->n_eval = n_eval;
    this->epsilon = epsilon;
    this->theta = theta_init;
    this->C = C;
    this->max_trials = max_trials;
    this->trust_region_radius = trust_region_radius;
    this->bobyqa_eval = bobyqa_eval;
    this->verbose = verbose;

    int d = bounds.size();
    this->theta = max(2.0 + 1.0 / (n_eval * d), theta_init);
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

private:
  int n_eval;
  double epsilon;
  double theta;
  int C;
  int max_trials;
  double trust_region_radius;
  int bobyqa_eval;
  bool verbose;
};