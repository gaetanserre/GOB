/*
 * Created in 2024 by Gaëtan Serré
 */

#include <iostream>
#include "utils.hh"
#include "python3.12/Python.h"
using namespace std;

class Optimizer
{
public:
  Optimizer(vec_bounds bounds, string name)
  {
    this->bounds = bounds;
    this->name = name;
    this->re.seed(time(NULL));
  };
  virtual double optimize(function<double(dyn_vector x)> f) = 0;

  double py_optimize(PyObject *f)
  {
    auto f_cpp = [f](dyn_vector x) -> double
    {
      PyObject *my_list = PyList_New(0);
      for (int i = 0; i < x.size(); i++)
      {
        PyList_Append(my_list, PyFloat_FromDouble(x[i]));
      }
      PyObject *result = PyObject_CallObject(f, my_list);
      return PyFloat_AsDouble(result);
    };
    return this->optimize(f_cpp);
  }

  vec_bounds bounds;
  string name;
  std::default_random_engine re;
};