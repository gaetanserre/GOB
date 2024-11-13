/*
 * Created in 2024 by Gaëtan Serré
 */

#include <iostream>
#include <random>
#include <chrono>
#include <eigen3/Eigen/Core>

typedef Eigen::VectorXd dyn_vector;
typedef std::vector<std::vector<double>> vec_bounds;

extern double max_vec(std::vector<double> &v);

extern double min_vec(std::vector<double> &v);

extern vec_bounds create_rect_bounds(double lb, double ub, int n);

extern double unif_random_double(std::default_random_engine &re, double lb, double ub);

extern dyn_vector unif_random_vector(std::default_random_engine &re, vec_bounds &bounds);

extern void print_vector(dyn_vector &x);