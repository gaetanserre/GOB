/*
 * Created in 2024 by Gaëtan Serré
 */

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
  virtual double optimize(function<double(dyn_vector x)> f) = 0;

  vec_bounds bounds;
  string name;
  std::default_random_engine re;
};