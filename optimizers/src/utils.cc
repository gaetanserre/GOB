/*
 * Created in 2024 by Gaëtan Serré
 */

#include "utils.hh"

std::vector<double> empty_vector()
{
  return std::vector<double>(0);
}

double max_vec(std::vector<double> &v)
{
  return *std::max_element(v.begin(), v.end());
}

double min_vec(std::vector<double> &v)
{
  return *std::min_element(v.begin(), v.end());
}

vec_bounds create_rect_bounds(double lb, double ub, int n)
{
  vec_bounds bounds(n, std::vector<double>(2));
  for (int i = 0; i < n; i++)
  {
    bounds[i][0] = lb;
    bounds[i][1] = ub;
  }
  return bounds;
}

double unif_random_double(std::default_random_engine &re, double lb, double ub)
{
  std::uniform_real_distribution<double> unif(lb, ub);
  double res = unif(re);
  re.seed(std::chrono::system_clock::now().time_since_epoch().count());
  return res;
}

dyn_vector unif_random_vector(std::default_random_engine &re, vec_bounds &bounds)
{
  int n = bounds.size();
  dyn_vector x(n);
  for (int i = 0; i < n; i++)
  {
    x(i) = unif_random_double(re, bounds[i][0], bounds[i][1]);
  }
  return x;
}

void print_vector(dyn_vector &x)
{
  std::cout << '[';
  for (int i = 0; i < x.size() - 1; i++)
  {
    std::cout << x(i) << ", ";
  }
  std::cout << x(x.size() - 1) << ']' << std::endl;
}

/* PyArrayObject *vector_to_nparray(dyn_vector &vec)
{

  // rows not empty
  if (vec.size() > 0)
  {

    npy_intp dims[1] = {vec.size()};

    PyArrayObject *vec_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *vec_array_pointer = (double *)PyArray_DATA(vec_array);

    std::copy(vec.begin(), vec.end(), vec_array_pointer);
    return vec_array;

    // no data at all
  }
  else
  {
    npy_intp dims[1] = {0};
    return (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
  }
} */

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

    std::copy(vec.data(), vec.data() + vec.size(), vec_array_pointer);
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