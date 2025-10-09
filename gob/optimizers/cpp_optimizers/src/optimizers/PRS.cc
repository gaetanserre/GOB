/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/PRS.hh"

result_eigen PRS::minimize(function<double(dyn_vector x)> f)
{
  int n = this->bounds.size();
  double min;
  dyn_vector best_sample;
  bool first = true;
  for (int i = 0; i < this->n_eval; i++)
  {
    dyn_vector x = unif_random_vector(this->re, this->bounds);
    double val = f(x);
    if (first || val < min)
    {
      min = val;
      best_sample = x;
      first = false;
    }

    if (this->has_stop_criterion && min < this->stop_criterion)
    {
      break;
    }
  }
  return {best_sample, min};
}