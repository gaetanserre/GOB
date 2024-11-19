/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include <deque>

class AdaLIPO_P : public Optimizer
{
public:
  AdaLIPO_P(vec_bounds bounds, int n_eval = 1000, int window_size = 5, double max_slope = 600) : Optimizer(bounds, "AdaLIPO+")
  {
    this->n_eval = n_eval;
    this->window_size = window_size;
    this->max_slope = max_slope;
  };

  virtual double minimize(function<double(dyn_vector x)> f);

  int n_eval;
  int window_size;
  double max_slope;

private:
  bool Bernoulli(double p);
  bool slope_stop_condition(deque<int> last_nb_samples);
};