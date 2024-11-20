#include <glpk.h>
#include <iostream>
#include "utils.hh"

void simplex(Eigen::MatrixXd M, glp_smcp *param)
{
  glp_prob *lp = glp_create_prob();
  glp_set_obj_dir(lp, GLP_MIN);
  int n_lambdas = M.cols();
  int n_relax = M.rows();
  int n_variables = n_lambdas + n_relax;

  // Create n_lambdas + n_relax variables
  glp_add_cols(lp, n_variables);

  for (int i = 1; i <= n_lambdas; i++)
  {
    glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
  }

  for (int i = n_lambdas + 1; i <= n_variables; i++)
  {
    glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, i, 1.0);
  }

  int n_constraints = M.rows();
  glp_add_rows(lp, n_constraints + 1);
  glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);

  for (int i = 2; i <= n_constraints + 1; i++)
  {
    glp_set_row_bnds(lp, i, GLP_UP, 0.0, 0.0);
  }

  int size_constraints_matrices = 1 + n_lambdas + (n_constraints * (n_lambdas + 1));

  int ia[size_constraints_matrices];
  int ja[size_constraints_matrices];
  double ar[size_constraints_matrices] = {0};

  for (int i = 1; i <= n_lambdas; i++)
  {
    ia[i] = 1;
    ja[i] = i;
    ar[i] = 1.0;
  }

  for (int i = 2; i <= n_constraints + 1; i++)
  {
    for (int j = 1; j <= n_lambdas; j++)
    {
      int idx = n_lambdas + (i - 2) * (n_lambdas + 1) + j;
      ia[idx] = i;
      ja[idx] = j;
      ar[idx] = M(i - 2, j - 1);
    }
    int idx = n_lambdas + (i - 2) * (n_lambdas + 1) + n_lambdas + 1;
    printf("idx: %d\n", idx);
    ia[idx] = i;
    ja[idx] = n_lambdas + i - 1;
    ar[idx] = -1.0;
  }
  // print ia
  std::cout << "ia: ";
  for (int i = 1; i < size_constraints_matrices; i++)
  {
    std::cout << ia[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "ja: ";
  for (int i = 1; i < size_constraints_matrices; i++)
  {
    std::cout << ja[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "ar: ";
  for (int i = 1; i < size_constraints_matrices; i++)
  {
    std::cout << ar[i] << " ";
  }
  std::cout << std::endl;

  printf("size: %d\n", size_constraints_matrices);
  glp_load_matrix(lp, size_constraints_matrices - 1, ia, ja, ar);

  glp_simplex(lp, param);

  double sum_relaxed = 0.0;
  for (int i = n_lambdas + 1; i <= n_variables; i++)
  {
    sum_relaxed += glp_get_col_prim(lp, i);
  }

  // print columns
  for (int i = 1; i <= n_variables; i++)
  {
    printf("x%d = %g\n", i, glp_get_col_prim(lp, i));
  }

  printf("sum_relaxed = %g\n", sum_relaxed);
}

int main()
{
  // Full ones matrix
  Eigen::MatrixXd M = Eigen::MatrixXd::Ones(2, 2);
  M(0, 1) = -2.0;
  M(1, 1) = -2.0;
  glp_smcp param;
  glp_init_smcp(&param);
  param.msg_lev = GLP_MSG_OFF;
  simplex(M, &param);
  return 0;
}
