/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"
#include "Simplex.hh"

class AdaRankOpt : public Optimizer
{
public:
  AdaRankOpt(vec_bounds bounds, int n_eval = 1000, double simplex_tol = 1e-6) : Optimizer(bounds, "AdaRankOpt")
  {
    this->n_eval = n_eval;
    this->simplex_tol = simplex_tol;
    this->param = new glp_smcp();
    glp_init_smcp(param);
    param->msg_lev = GLP_MSG_OFF;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_eval;
  double simplex_tol;
  glp_smcp *param;

private:
  static Eigen::MatrixXd polynomial_matrix(vector<dyn_vector> &X, int degree);
  bool is_polyhedral_set_empty(vector<dyn_vector> &X, int degree);
};