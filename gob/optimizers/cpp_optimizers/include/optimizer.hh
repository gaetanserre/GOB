/*
 * Created in 2024 by Gaëtan Serré
 */

#pragma once

#include <iostream>
#include "utils.hh"
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
  ~Optimizer()
  {
    Py_Finalize();
  };
  virtual result_eigen minimize(function<double(dyn_vector x)> f) = 0;

  result py_minimize(PyObject *f);

  vec_bounds bounds;
  string name;
  std::default_random_engine re;
};