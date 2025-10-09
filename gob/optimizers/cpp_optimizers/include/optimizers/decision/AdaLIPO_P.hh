/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"
#include <deque>

class AdaLIPO_P : public Optimizer
{
public:
  AdaLIPO_P(vec_bounds bounds,
            int n_eval = 1000,
            int max_trials = 800,
            double trust_region_radius = 0.1,
            int bobyqa_eval = 10) : Optimizer(bounds, "AdaLIPO+TR")
  {
    this->n_eval = n_eval;
    this->max_trials = max_trials;
    this->trust_region_radius = trust_region_radius;
    this->bobyqa_eval = bobyqa_eval;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

private:
  int n_eval;
  int max_trials;
  double trust_region_radius;
  int bobyqa_eval;
};