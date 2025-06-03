/*
 * Created in 2024 by Gaëtan Serré
 */

#include "utils.hh"

vector<double> empty_vector()
{
  return vector<double>(0);
}

double max_vec(const vector<double> &v)
{
  return *max_element(v.begin(), v.end());
}

int argmax_vec(const vector<double> &v)
{
  return distance(v.begin(), max_element(v.begin(), v.end()));
}

double min_vec(const vector<double> &v)
{
  return *min_element(v.begin(), v.end());
}

int argmin_vec(const vector<double> &v)
{
  return distance(v.begin(), min_element(v.begin(), v.end()));
}

vec_bounds create_rect_bounds(double lb, double ub, int n)
{
  vec_bounds bounds(n, vector<double>(2));
  for (int i = 0; i < n; i++)
  {
    bounds[i][0] = lb;
    bounds[i][1] = ub;
  }
  return bounds;
}

double unif_random_double(mt19937_64 &re, double lb, double ub)
{
  uniform_real_distribution<double> unif(lb, ub);
  return unif(re);
}

double normal_random_double(mt19937_64 &re, double mean, double stddev)
{
  normal_distribution<double> norm(mean, stddev);
  return norm(re);
}

dyn_vector normal_random_vector(mt19937_64 &re, int size, double mean, double stddev)
{
  dyn_vector x(size);
  for (int i = 0; i < size; i++)
  {
    x(i) = normal_random_double(re, mean, stddev);
  }
  return x;
}

dyn_vector unif_random_vector(mt19937_64 &re, vec_bounds &bounds)
{
  int n = bounds.size();
  dyn_vector x(n);
  for (int i = 0; i < n; i++)
  {
    x(i) = unif_random_double(re, bounds[i][0], bounds[i][1]);
  }
  return x;
}

void print_vector(const dyn_vector &x)
{
  cout << '[';
  for (int i = 0; i < x.size() - 1; i++)
  {
    cout << x(i) << ", ";
  }
  cout << x(x.size() - 1) << ']' << endl;
}

void print_matrix(const Eigen::MatrixXd &M)
{
  cout << '[' << endl;
  for (int i = 0; i < M.rows(); i++)
  {
    dyn_vector row = M.row(i);
    print_vector(row);
  }
  cout << ']' << endl;
}

PyArrayObject *vector_to_nparray(const dyn_vector &vec)
{

  if (vec.size() == 0)
  {
    npy_intp dims[1] = {0};
    return (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
  }
  else
  {
    npy_intp dims[1] = {vec.size()};

    PyArrayObject *vec_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *vec_array_pointer = (double *)PyArray_DATA(vec_array);

    copy(vec.data(), vec.data() + vec.size(), vec_array_pointer);
    return vec_array;
  }
}

void py_init()
{
  Py_Initialize();
  _import_array();
  return;
}

void py_finalize()
{
  Py_Finalize();
}

dyn_vector sub_vector(dyn_vector v, const unsigned int &start, const unsigned int &end)
{
  return v.segment(start, end - start);
}

bool Bernoulli(mt19937_64 &re, double p)
{
  bernoulli_distribution d(p);
  return d(re);
}

dyn_vector clip_vector(dyn_vector x, vec_bounds &bounds)
{
  dyn_vector res = x;
  for (int i = 0; i < x.size(); i++)
  {
    res(i) = max(bounds[i][0], min(bounds[i][1], x(i)));
  }
  return res;
}

bool contains_nan(const dyn_vector &x)
{
  for (int i = 0; i < x.size(); i++)
  {
    if (isnan(x(i)))
      return true;
  }
  return false;
}