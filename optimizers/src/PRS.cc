/*
 * Created in 2024 by Gaëtan Serré
 */

#include "PRS.hh"

double PRS::optimize(function<double(dyn_vector x)> f)
{
  int n = this->bounds.size();
  double min;
  bool first = true;
  for (int i = 0; i < this->n_eval; i++)
  {
    dyn_vector x = unif_random_vector(this->re, this->bounds);
    double val = f(x);
    if (first || val < min)
    {
      min = val;
      first = false;
    }
  }
  return min;
}