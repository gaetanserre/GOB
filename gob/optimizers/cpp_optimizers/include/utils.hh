/*
 * Created in 2024 by Gaëtan Serré
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Core>
#include "numpy/ndarrayobject.h"
#include "Python.h"
using namespace std;

typedef Eigen::VectorXd dyn_vector;
typedef vector<vector<double>> vec_bounds;
typedef pair<vector<double>, double> result;
typedef pair<dyn_vector, double> result_eigen;

extern vector<double> empty_vector();

extern double max_vec(const vector<double> &v);

extern double min_vec(const vector<double> &v);

extern int argmax_vec(const vector<double> &v);

extern int argmin_vec(const vector<double> &v);

extern vec_bounds create_rect_bounds(double lb, double ub, int n);

extern double unif_random_double(mt19937_64 &re, double lb, double ub);

extern double normal_random_double(mt19937_64 &re, double mean, double stddev);

extern dyn_vector unif_random_vector(mt19937_64 &re, vec_bounds &bounds);

extern dyn_vector normal_random_vector(mt19937_64 &re, int size, double mean, double stddev);

extern void print_vector(const dyn_vector &x);

extern void print_matrix(const Eigen::MatrixXd &M);

extern PyArrayObject *vector_to_nparray(const dyn_vector &vec);

extern void py_init();

extern void py_finalize();

extern dyn_vector sub_vector(dyn_vector v, const unsigned int &start, const unsigned int &end);

extern bool Bernoulli(mt19937_64 &re, double p);

extern dyn_vector clip_vector(dyn_vector x, vec_bounds &bounds);

extern bool contains_nan(const dyn_vector &x);