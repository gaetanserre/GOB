from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

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
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/AdaLIPO_P.hh":
  cdef cppclass CAdaLIPO_P "AdaLIPO_P":
    CAdaLIPO_P(
      vector[vector[double]] bounds,
      int n_eval,
      int max_trials,
      double trust_region_radius,
      int bobyqa_eval
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/CMA_ES.hh":
  cdef cppclass CCMA_ES "CMA_ES":
    CCMA_ES(vector[vector[double]] bounds, int n_eval, vector[double] m0, double sigma)
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/SBS.hh":
  cdef cppclass CSBS "SBS":
    CSBS(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      int k,
      double sigma,
      double lr
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/CBO.hh":
  cdef cppclass CCBO "CBO":
    CCBO(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double lam,
      double epsilon,
      double beta,
      double sigma,
      bool use_batch
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/AdaRankOpt.hh":
  cdef cppclass CAdaRankOpt "AdaRankOpt":
    CAdaRankOpt(
      vector[vector[double]] bounds,
      int n_eval,
      int max_trials,
      int max_degree,
      double trust_region_radius,
      int bobyqa_eval
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/ECP.hh":
  cdef cppclass CECP "ECP":
    CECP(
      vector[vector[double]] bounds,
      int n_eval,
      double epsilon,
      double theta_init,
      int C,
      int max_trials,
      double trust_region_radius,
      int bobyqa_eval
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

# Python interface

cdef class PRS:
  cdef CPRS *thisptr
  def __cinit__(self, bounds, int n_eval=1000):
    self.thisptr = new CPRS(bounds, n_eval)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class AdaLIPO_P:
  cdef CAdaLIPO_P *thisptr
  def __cinit__(
      self,
      bounds,
      int n_eval=1000,
      int max_trials=800,
      double trust_region_radius=0.1,
      int bobyqa_eval=10
    ):
    self.thisptr = new CAdaLIPO_P(
        bounds,
        n_eval,
        max_trials,
        trust_region_radius,
        bobyqa_eval)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class CMA_ES:
  cdef CCMA_ES *thisptr
  def __cinit__(self, bounds, int n_eval=1000, vector[double] m0=[], double sigma=0.1):
    self.thisptr = new CCMA_ES(bounds, n_eval, m0, sigma)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class SBS:
  cdef CSBS *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles=200,
    int iter=100,
    int k=10_000,
    double sigma=0.01,
    double lr=0.5
  ):
    self.thisptr = new CSBS(bounds, n_particles, iter, k, sigma, lr)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class CBO:
  cdef CCBO *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles=200,
    int iter=100,
    double lam=1e-1,
    double epsilon=1e-2,
    double beta=5,
    double sigma=5,
    bool use_batch=True
  ):
    self.thisptr = new CCBO(bounds, n_particles, iter, lam, epsilon, beta, sigma, use_batch)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class AdaRankOpt:
  cdef CAdaRankOpt *thisptr
  def __cinit__(
      self,
      bounds,
      int n_eval=1000,
      int max_trials=800,
      int max_degree=80,
      double trust_region_radius=0.1,
      int bobyqa_eval=10
    ):
    self.thisptr = new CAdaRankOpt(
        bounds,
        n_eval,
        max_trials,
        max_degree,
        trust_region_radius,
        bobyqa_eval)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

  def __del__(self):
    del self.thisptr

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)

cdef class ECP:
  cdef CECP *thisptr
  def __cinit__(
      self,
      bounds,
      int n_eval=50,
      double epsilon=1e-2,
      double theta_init=1.001,
      int C =1000,
      int max_trials=1_000_000,
      double trust_region_radius=0.1,
      int bobyqa_eval=10
    ):
    self.thisptr = new CECP(
        bounds,
        n_eval,
        epsilon,
        theta_init,
        C,
        max_trials,
        trust_region_radius,
        bobyqa_eval)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    return self.thisptr.py_minimize(pyob_ptr)
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

  def __del__(self):
    del self.thisptr

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)