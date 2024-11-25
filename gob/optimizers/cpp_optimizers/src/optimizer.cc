/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"

result Optimizer::py_minimize(PyObject *f)
{
  auto f_cpp = [f](dyn_vector x) -> double
  {
    PyObject *my_list = (PyObject *)vector_to_nparray(x);
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, my_list);
    PyObject *result = PyObject_CallObject(f, args);
    return PyFloat_AsDouble(result);
  };
  return this->minimize(f_cpp);
}