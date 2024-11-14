/*
 * Created in 2024 by Gaëtan Serré
 */

#pragma once

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

  double py_optimize(PyObject *f);

  vec_bounds bounds;
  string name;
  std::default_random_engine re;
};