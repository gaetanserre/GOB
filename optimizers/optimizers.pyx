# FFI for the GKLS library

from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from eigency.core cimport *

ctypedef double (*f_type)(VectorXd)

cdef extern from "include/utils.hh":
  vector[vector[double]] create_rect_bounds_ "create_rect_bounds"(double lb, double ub, int n)

cdef extern from "include/PRS.hh":
  cdef cppclass PRS:
    PRS(vector[vector[double]] bounds, int n_eval)
    void py_optimize(PyObject* f)

# Python interface

cdef class PRSWrapper:
     cdef PRS *thisptr
     def __cinit__(self, bounds, int n_eval):
        self.thisptr = new PRS(bounds, n_eval)
     def optimize(self, f):
        cdef PyObject* pyob_ptr = <PyObject*>f
        self.thisptr.py_optimize(pyob_ptr)

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)
