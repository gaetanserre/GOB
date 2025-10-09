/*
 * Created in 2025 by Gaëtan Serré
 */

#include "utils.hh"

extern Eigen::MatrixXd zero_noise(const int &rows, const int &cols);

extern Eigen::MatrixXd normal_noise(const int &rows, const int &cols, std::mt19937_64 re, double mean = 0, double stddev = 1);