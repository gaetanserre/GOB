/*
 * Created in 2024 by Gaëtan Serré
 */

#include "glpk.h"
#include "utils.hh"

extern int simplex(Eigen::MatrixXd M, glp_smcp *param);