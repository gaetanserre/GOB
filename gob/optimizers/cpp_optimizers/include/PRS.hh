/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"

class PRS : public Optimizer
{
public:
  PRS(vec_bounds bounds, int n_eval = 1000) : Optimizer(bounds, "PRS")
  {
    this->n_eval = n_eval;
  };

  virtual double minimize(function<double(dyn_vector x)> f);

  int n_eval;
};