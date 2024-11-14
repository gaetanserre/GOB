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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/ndarrayobject.h"

int main()
{
  Py_Initialize();
  _import_array();
  npy_intp dims[3] = {2, 2, 2};
  PyObject *r_obj_array = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
  // Py_Finalize();
  return 0;
}