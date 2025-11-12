/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"

class PRS : public Optimizer
{
public:
  PRS(vec_bounds bounds, int n_eval) : Optimizer(bounds, "PRS")
  {
    this->n_eval = n_eval;
  }

  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  int n_eval;
};