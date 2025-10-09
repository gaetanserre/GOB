/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"

class CMA_ES : public Optimizer
{
public:
  CMA_ES(
      vec_bounds bounds,
      int n_eval,
      std::vector<double> m0 = empty_vector(),
      double sigma = 0.1) : Optimizer(bounds, "CMA-ES")
  {
    this->n_eval = n_eval;
    this->m0 = m0;
    this->sigma = sigma;

    this->transform_bounds(bounds);
  }

  ~CMA_ES()
  {
    delete[] lbounds;
    delete[] ubounds;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

private:
  virtual void transform_bounds(vec_bounds bounds);
  int n_eval;
  double *lbounds;
  double *ubounds;
  std::vector<double> m0;
  double sigma;
};