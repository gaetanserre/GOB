/* #include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

int print_eigen(Eigen::MatrixX3d m)
{
    // Eigen Matrices do have rule to print them with std::cout
    std::cout << m << std::endl;
    return 0;
}

int main()
{
    Eigen::Matrix3d test; //3 by 3 double precision matrix initialization

    // Let's make it a symmetric matrix
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
            test(i,j) = (i+1)*(1+j);
    }

    // Print
    print_eigen(test);

    return 0;
} */

#include "AdaLIPO_P.hh"
#include <boost/python.hpp>
#include <Python.h>

int main()
{
  vec_bounds bounds = create_rect_bounds(-1, 1, 2);
  AdaLIPO_P prs(bounds);
  PyObject *pFunc = NULL;

  auto f = [](dyn_vector x) -> double
  {
    return x.transpose() * x;
  };

  std::cout << prs.optimize(f) << std::endl;

  return 0;
}