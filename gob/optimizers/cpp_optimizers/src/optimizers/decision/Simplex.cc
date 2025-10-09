#include "optimizers/decision/Simplex.hh"

int simplex(Eigen::MatrixXd M, glp_smcp *param)
{
  glp_prob *lp = glp_create_prob();
  glp_set_obj_dir(lp, GLP_MIN);
  int n_variables = M.cols();

  // Create variables
  glp_add_cols(lp, n_variables);

  for (int i = 1; i <= n_variables; i++)
  {
    glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, i, 1.0);
  }

  int n_constraints = M.rows();
  glp_add_rows(lp, n_constraints + 1);
  glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);

  for (int i = 2; i <= n_constraints + 1; i++)
  {
    glp_set_row_bnds(lp, i, GLP_FX, 0.0, 0.0);
  }

  int size_constraints_matrices = 1 + n_variables + (n_constraints * n_variables);

  int *ia = new int[size_constraints_matrices];
  int *ja = new int[size_constraints_matrices];
  double *ar = new double[size_constraints_matrices];

  for (int i = 1; i <= n_variables; i++)
  {
    ia[i] = 1;
    ja[i] = i;
    ar[i] = 1.0;
  }

  for (int i = 2; i <= n_constraints + 1; i++)
  {
    for (int j = 1; j <= n_variables; j++)
    {
      int idx = n_variables + (i - 2) * n_variables + j;
      ia[idx] = i;
      ja[idx] = j;
      ar[idx] = M(i - 2, j - 1);
    }
  }

  glp_load_matrix(lp, size_constraints_matrices - 1, ia, ja, ar);

  glp_simplex(lp, param);
  int status = glp_get_status(lp);
  glp_delete_prob(lp);
  glp_free_env();
  delete[] ia;
  delete[] ja;
  delete[] ar;

  return status;
}
