/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/optimizer.hh"

result Optimizer::py_minimize(PyObject *f)
{
  auto f_cpp = [f](dyn_vector x) -> double
  {
    PyObject *my_list = (PyObject *)vector_to_nparray(x);
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, my_list);
    PyObject *result = PyObject_CallObject(f, args);
    double value = 0.0;
    if (result)
    {
      value = PyFloat_AsDouble(result);
      Py_DECREF(result);
    }
    else
    {
      PyErr_Print();
    }
    Py_DECREF(args);
    return value;
  };
  result_eigen res = this->minimize(f_cpp);
  vector<double> x(res.first.data(), res.first.data() + res.first.size());
  return {x, res.second};
}