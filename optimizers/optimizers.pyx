# FFI for the GKLS library

from libcpp.vector cimport vector
from eigency.core import *

ctypedef double (*f_type)(VectorXd)

cdef extern from "include/utils.hh":
  vector[vector[double]] create_rect_bounds_ "create_rect_bounds"(double lb, double ub, int n)

cdef extern from "include/PRS.hh":
  cdef cppclass PRS:
    PRS(vector[vector[double]] bounds, int n_eval)
    void optimize(f_type f)

# Python interface

# cdef class PRSWrapper:
#     cdef PRS *thisptr
#     def __cinit__(self, bounds, int n_eval):
#         self.thisptr = new PRS(bounds, n_eval)
#     def __dealloc__(self):
#         del self.thisptr
#     def optimize(self, f):
#         self.thisptr.optimize(f)


def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)
