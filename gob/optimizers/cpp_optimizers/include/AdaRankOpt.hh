/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include "bobyqa.hh"
#include "Simplex.hh"

class AdaRankOpt : public Optimizer
{
public:
  AdaRankOpt(vec_bounds bounds, int n_eval = 1000, int max_degree = 40, int max_samples = 10000, bool bobyqa = true, int bobyqa_maxfun = 100, bool verbose = false) : Optimizer(bounds, "AdaRankOpt")
  {
    this->n_eval = n_eval;
    this->max_degree = max_degree;
    this->max_samples = max_samples;
    this->bobyqa = bobyqa;
    this->bobyqa_maxfun = bobyqa_maxfun;
    this->verbose = verbose;

    this->param = new glp_smcp();
    glp_init_smcp(param);
    param->msg_lev = GLP_MSG_OFF;
    param->it_lim = 100;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  int max_degree;
  int max_samples;
  bool bobyqa;
  int bobyqa_maxfun;
  bool verbose;
  glp_smcp *param;

private:
  static Eigen::MatrixXd polynomial_matrix(vector<pair<dyn_vector, double>> &samples, int degree);
  bool is_polyhedral_set_empty(vector<pair<dyn_vector, double>> &samples, int degree);
};