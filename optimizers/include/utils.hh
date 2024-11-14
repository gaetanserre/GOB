/*
 * Created in 2024 by Gaëtan Serré
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <random>
#include <chrono>
#include <eigen3/Eigen/Core>
#include "numpy/ndarrayobject.h"

typedef Eigen::VectorXd dyn_vector;
typedef std::vector<std::vector<double>> vec_bounds;

extern std::vector<double> empty_vector();

extern double max_vec(std::vector<double> &v);

extern double min_vec(std::vector<double> &v);

extern vec_bounds create_rect_bounds(double lb, double ub, int n);

extern double unif_random_double(std::default_random_engine &re, double lb, double ub);

extern dyn_vector unif_random_vector(std::default_random_engine &re, vec_bounds &bounds);

extern void print_vector(dyn_vector &x);

extern PyArrayObject *vector_to_nparray(const dyn_vector &vec);

extern void py_init();

extern void py_finalize();