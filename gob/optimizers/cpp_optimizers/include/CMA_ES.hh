/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"

class CMA_ES : public Optimizer
{
public:
  CMA_ES(vec_bounds bounds, int n_eval, std::vector<double> m0 = empty_vector(), double sigma = 0.1) : Optimizer(bounds, "CMA-ES")
  {
    this->n_eval = n_eval;
    this->m0 = m0;
    this->sigma = sigma;
  };

  virtual double minimize(function<double(dyn_vector x)> f);

  int n_eval;
  std::vector<double> m0;
  double sigma;
};