/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include "Simplex.hh"

class AdaRankOpt : public Optimizer
{
public:
  AdaRankOpt(
      vec_bounds bounds,
      int n_eval = 1000,
      int max_samples = 800,
      int max_degree = 80,
      double trust_region_radius = 0.1,
      int bobyqa_eval = 10,
      bool verbose = false) : Optimizer(bounds, "AdaRankOpt")
  {
    this->n_eval = n_eval;
    this->max_samples = max_samples;
    this->max_degree = max_degree;
    this->trust_region_radius = trust_region_radius;
    this->bobyqa_eval = bobyqa_eval;
    this->verbose = verbose;

    this->param = new glp_smcp();
    glp_init_smcp(param);
    param->msg_lev = GLP_MSG_OFF;
    param->it_lim = 100;
  };

  ~AdaRankOpt()
  {
    delete param;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  int max_samples;
  int max_degree;
  double trust_region_radius;
  int bobyqa_eval;
  bool verbose;
  glp_smcp *param;

private:
  static Eigen::MatrixXd polynomial_matrix(vector<pair<dyn_vector, double>> &samples, int degree);
};