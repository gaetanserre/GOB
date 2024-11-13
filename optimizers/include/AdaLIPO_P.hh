/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include <deque>

class AdaLIPO_P : public Optimizer
{
public:
  AdaLIPO_P(vec_bounds bounds, int n_eval = 1000, int window_slope = 5, double max_float = 600) : Optimizer(bounds, "AdaLIPO+")
  {
    this->n_eval = n_eval;
    this->window_slope = window_slope;
    this->max_float = max_float;
  };

  virtual double optimize(function<double(dyn_vector x)> f);

  int n_eval;
  int window_slope;
  double max_float;

private:
  bool Bernoulli(double p);
  bool slope_stop_condition(std::deque<float> last_nb_samples);
};