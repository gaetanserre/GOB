/*
 * Created in 2024 by Gaëtan Serré
 */

#include "glpk.h"
#include "utils.hh"

extern bool simplex(Eigen::MatrixXd M, glp_smcp *param, double tol);