/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizer.hh"
#include <deque>

class ECP : public Optimizer
{
public:
  ECP(vec_bounds bounds,
      int n_eval = 50,
      double epsilon = 1e-2,
      double theta_init = 1.001,
      int C = 1000,
      bool verbose = false) : Optimizer(bounds, "ECP")
  {
    this->n_eval = n_eval;
    this->epsilon = epsilon;
    this->theta = theta_init;
    this->C = C;
    this->verbose = verbose;

    int d = bounds.size();
    this->theta = max(1.0 + 1.0 / (n_eval * d), theta_init);
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  double epsilon;
  double theta;
  int C;
  bool verbose;
};