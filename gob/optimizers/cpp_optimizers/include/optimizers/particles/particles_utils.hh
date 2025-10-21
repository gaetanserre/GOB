/*
 * Created in 2025 by Gaëtan Serré
 */

#include "utils.hh"

extern double log_sum_exp(double *begin, double *end);

extern dyn_vector compute_consensus(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals, double &beta);

extern Eigen::MatrixXd pairwise_dist(const Eigen::MatrixXd &particles);

extern Eigen::MatrixXd rbf(const Eigen::MatrixXd &particles, const double &sigma);