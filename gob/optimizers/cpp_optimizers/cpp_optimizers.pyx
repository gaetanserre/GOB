from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "include/utils.hh":
  vector[vector[double]] create_rect_bounds_ "create_rect_bounds"(double lb, double ub, int n)

cdef extern from "include/utils.hh":
  void py_init()

cdef extern from "include/utils.hh":
  void py_finalize()

cdef extern from "include/PRS.hh":
  cdef cppclass CPRS "PRS":
    CPRS(vector[vector[double]] bounds, int n_eval)
    pair[vector[double], double] py_minimize(PyObject* f)

cdef extern from "include/AdaLIPO_P.hh":
  cdef cppclass CAdaLIPO_P "AdaLIPO_P":
    CAdaLIPO_P(vector[vector[double]] bounds, int n_eval, int window_size, double max_slope)
    pair[vector[double], double] py_minimize(PyObject* f)

cdef extern from "include/CMA_ES.hh":
  cdef cppclass CCMA_ES "CMA_ES":
    CCMA_ES(vector[vector[double]] bounds, int n_eval, vector[double] m0, double sigma)
    pair[vector[double], double] py_minimize(PyObject* f)

cdef extern from "include/SBS.hh":
  cdef cppclass CSBS "SBS":
    CSBS(
      vector[vector[double]] bounds,
      int n_particles,
      int svgd_iter,
      vector[int] k_iter,
      double sigma,
      double lr
    )
    pair[vector[double], double] py_minimize(PyObject* f)

cdef extern from "include/AdaRankOpt.hh":
  cdef cppclass CAdaRankOpt "AdaRankOpt":
    CAdaRankOpt(
      vector[vector[double]] bounds,
      int n_eval,
      double simplex_tol
    )
    pair[vector[double], double] py_minimize(PyObject* f)

# Python interface

cdef class PRS:
  cdef CPRS *thisptr
  def __cinit__(self, bounds, int n_eval=1000):
    self.thisptr = new CPRS(bounds, n_eval)
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

cdef class AdaLIPO_P:
  cdef CAdaLIPO_P *thisptr
  def __cinit__(self, bounds, int n_eval=1000, int window_size=5, double max_slope=600):
    self.thisptr = new CAdaLIPO_P(bounds, n_eval, window_size, max_slope)
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

cdef class CMA_ES:
  cdef CCMA_ES *thisptr
  def __cinit__(self, bounds, int n_eval=1000, vector[double] m0=[], double sigma=0.1):
    self.thisptr = new CCMA_ES(bounds, n_eval, m0, sigma)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

cdef class SBS:
  cdef CSBS *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles=200,
    int svgd_iter=100,
    vector[int] k_iter=[10_000],
    double sigma=0.01,
    double lr=0.5
  ):
    self.thisptr = new CSBS(bounds, n_particles, svgd_iter, k_iter, sigma, lr)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

cdef class AdaRankOpt:
  cdef CAdaRankOpt *thisptr
  def __cinit__(self, bounds, int n_eval=1000, double simplex_tol=1e-6):
    self.thisptr = new CAdaRankOpt(bounds, n_eval, simplex_tol)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)