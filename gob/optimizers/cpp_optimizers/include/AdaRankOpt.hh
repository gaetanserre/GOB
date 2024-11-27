/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include "Simplex.hh"

class AdaRankOpt : public Optimizer
{
public:
  AdaRankOpt(vec_bounds bounds, int n_eval = 1000, int max_degree = 40, int max_tries = 10000, bool verbose = false) : Optimizer(bounds, "AdaRankOpt")
  {
    this->n_eval = n_eval;
    this->max_degree = max_degree;
    this->max_tries = max_tries;
    this->verbose = verbose;

    this->param = new glp_smcp();
    glp_init_smcp(param);
    param->msg_lev = GLP_MSG_OFF;
    param->it_lim = 10000;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  int max_degree;
  int max_tries;
  bool verbose;
  glp_smcp *param;

private:
  static Eigen::MatrixXd polynomial_matrix(vector<dyn_vector> &X, int degree);
  bool is_polyhedral_set_empty(vector<dyn_vector> &X, int degree);
};