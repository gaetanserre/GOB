from cpython.ref cimport PyObject
from libcpp.vector cimport vector

cdef extern from "include/utils.hh":
  vector[vector[double]] create_rect_bounds_ "create_rect_bounds"(double lb, double ub, int n)

cdef extern from "include/utils.hh":
  void py_init()

cdef extern from "include/utils.hh":
  void py_finalize()

cdef extern from "include/PRS.hh":
  cdef cppclass CPRS "PRS":
    CPRS(vector[vector[double]] bounds, int n_eval)
    double py_optimize(PyObject* f)

cdef extern from "include/AdaLIPO_P.hh":
  cdef cppclass CAdaLIPO_P "AdaLIPO_P":
    CAdaLIPO_P(vector[vector[double]] bounds, int n_eval, int window_size, double max_slope)
    double py_optimize(PyObject* f)

cdef extern from "include/CMA_ES.hh":
  cdef cppclass CCMA_ES "CMA_ES":
    CCMA_ES(vector[vector[double]] bounds, int n_eval, vector[double] m0, double sigma)
    double py_optimize(PyObject* f)
    str get_name()

# Python interface

cdef class PRS:
  cdef CPRS *thisptr
  def __cinit__(self, bounds, int n_eval=1000):
    self.thisptr = new CPRS(bounds, n_eval)
  def optimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_optimize(pyob_ptr)

cdef class AdaLIPO_P:
  cdef CAdaLIPO_P *thisptr
  def __cinit__(self, bounds, int n_eval=1000, int window_size=5, double max_slope=600):
    self.thisptr = new CAdaLIPO_P(bounds, n_eval, window_size, max_slope)
  def optimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_optimize(pyob_ptr)

cdef class CMA_ES:
  cdef CCMA_ES *thisptr
  def __cinit__(self, bounds, int n_eval=1000, vector[double] m0=[], double sigma=0.1):
    self.thisptr = new CCMA_ES(bounds, n_eval, m0, sigma)

  def optimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_optimize(pyob_ptr)

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)
