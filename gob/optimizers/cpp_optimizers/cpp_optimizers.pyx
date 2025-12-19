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

cdef extern from "include/optimizers/PRS.hh":
  cdef cppclass CPRS "PRS":
    CPRS(vector[vector[double]] bounds, int n_eval)
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/decision/AdaLIPO_P.hh":
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

cdef extern from "include/optimizers/CMA_ES.hh":
  cdef cppclass CCMA_ES "CMA_ES":
    CCMA_ES(vector[vector[double]] bounds, int n_eval, vector[double] m0, double sigma)
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/SBS.hh":
  cdef cppclass CSBS "SBS":
    CSBS(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double sigma,
      int batch_size
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/CBO.hh":
  cdef cppclass CCBO "CBO":
    CCBO(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double lam,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      int batch_size
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/decision/AdaRankOpt.hh":
  cdef cppclass CAdaRankOpt "AdaRankOpt":
    CAdaRankOpt(
      vector[vector[double]] bounds,
      int n_eval,
      int max_trials,
      int max_degree,
      double trust_region_radius,
      int bobyqa_eval,
      int it_lim
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/decision/ECP.hh":
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

cdef extern from "include/optimizers/particles/PSO.hh":
  cdef cppclass CPSO "PSO":
    CPSO(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double omega,
      double c2,
      double beta,
      double alpha,
      int batch_size
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/Langevin.hh":
  cdef cppclass CLangevin "Langevin":
    CLangevin(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      int batch_size
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/SMD/SMD_Langevin.hh":
  cdef cppclass CSMD_Langevin "SMD_Langevin":
    CSMD_Langevin(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      double gamma,
      double lambda_,
      double delta,
      int moment,
      bool independent_noise
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/SMD/SMD_SBS.hh":
  cdef cppclass CSMD_SBS "SMD_SBS":
    CSMD_SBS(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double sigma,
      double gamma,
      double lambda_,
      double delta,
      int moment
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/SMD/SMD_CBO.hh":
  cdef cppclass CSMD_CBO "SMD_CBO":
    CSMD_CBO(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double lambda_,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      double gamma,
      double lambda_cn,
      double delta,
      int moment,
      bool independent_noise
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/Full_Noise.hh":
  cdef cppclass CFull_Noise "Full_Noise":
    CFull_Noise(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double alpha,
      int batch_size
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/GCN/GCN_CBO.hh":
  cdef cppclass CGCN_CBO "GCN_CBO":
    CGCN_CBO(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double lambda_,
      double epsilon,
      double beta,
      double sigma,
      double alpha,
      double sigma_cn,
      bool independent_noise
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/GCN/GCN_SBS.hh":
  cdef cppclass CGCN_SBS "GCN_SBS":
    CGCN_SBS(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double sigma,
      double sigma_cn,
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

cdef extern from "include/optimizers/particles/common-noise/GCN/GCN_Langevin.hh":
  cdef cppclass CGCN_Langevin "GCN_Langevin":
    CGCN_Langevin(
      vector[vector[double]] bounds,
      int n_particles,
      int iter,
      double dt,
      double beta,
      double sigma_cn,
      bool independent_noise
    )
    pair[vector[double], double] py_minimize(PyObject* f)
    void set_stop_criterion(double stop_criterion)

# Python interface

cdef class PRS:
  cdef CPRS *thisptr
  def __cinit__(self, bounds, int n_eval):
    self.thisptr = new CPRS(bounds, n_eval)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class AdaLIPO_P:
  cdef CAdaLIPO_P *thisptr
  def __cinit__(
      self,
      bounds,
      int n_eval,
      int max_trials,
      double trust_region_radius,
      int bobyqa_eval
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
    res = self.thisptr.py_minimize(pyob_ptr)
    return res
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class CMA_ES:
  cdef CCMA_ES *thisptr
  def __cinit__(self, bounds, int n_eval, vector[double] m0, double sigma):
    self.thisptr = new CCMA_ES(bounds, n_eval, m0, sigma)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class SBS:
  cdef CSBS *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double sigma,
    int batch_size
  ):
    self.thisptr = new CSBS(bounds, n_particles, iter, dt, sigma, batch_size)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class CBO:
  cdef CCBO *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double lam,
    double epsilon,
    double beta,
    double sigma,
    double alpha,
    int batch_size
  ):
    self.thisptr = new CCBO(bounds, n_particles, iter, dt, lam, epsilon, beta, sigma, alpha, batch_size)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class AdaRankOpt:
  cdef CAdaRankOpt *thisptr
  def __cinit__(
      self,
      bounds,
      int n_eval,
      int max_trials,
      int max_degree,
      double trust_region_radius,
      int bobyqa_eval,
      int it_lim
    ):
    self.thisptr = new CAdaRankOpt(
        bounds,
        n_eval,
        max_trials,
        max_degree,
        trust_region_radius,
        bobyqa_eval,
        it_lim)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res
  
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
      int n_eval,
      double epsilon,
      double theta_init,
      int C,
      int max_trials,
      double trust_region_radius,
      int bobyqa_eval
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
    res = self.thisptr.py_minimize(pyob_ptr)
    return res
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

  def __del__(self):
    del self.thisptr

def create_rect_bounds(lb, ub, n):
    return create_rect_bounds_(lb, ub, n)

cdef class PSO:
  cdef CPSO *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double omega,
    double c2,
    double beta,
    double alpha,
    int batch_size
  ):
    self.thisptr = new CPSO(bounds, n_particles, iter, dt, omega, c2, beta, alpha, batch_size)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr
    
cdef class Langevin:
  cdef CLangevin *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    beta,
    int batch_size
  ):
    self.thisptr = new CLangevin(bounds, n_particles, iter, dt, beta, batch_size)
  
  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)
  
  def __del__(self):
    del self.thisptr

cdef class SMD_Langevin:
  cdef CSMD_Langevin *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double beta,
    double gamma,
    double lambda_,
    double delta,
    int moment,
    bool independent_noise
  ):
    self.thisptr = new CSMD_Langevin(bounds, n_particles, iter, dt, beta, gamma, lambda_, delta, moment, independent_noise)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class SMD_SBS:
  cdef CSMD_SBS *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double sigma,
    double gamma,
    double lambda_,
    double delta,
    int moment
  ):
    self.thisptr = new CSMD_SBS(bounds, n_particles, iter, dt, sigma, gamma, lambda_, delta, moment)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class SMD_CBO:
  cdef CSMD_CBO *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double lambda_,
    double epsilon,
    double beta,
    double sigma,
    double alpha,
    double gamma,
    double lambda_cn,
    double delta,
    int moment,
    bool independent_noise
  ):
    self.thisptr = new CSMD_CBO(bounds, n_particles, iter, dt, lambda_, epsilon, beta, sigma, alpha, gamma, lambda_cn, delta, moment, independent_noise)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class Full_Noise:
  cdef CFull_Noise *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double alpha,
    int batch_size
  ):
    self.thisptr = new CFull_Noise(bounds, n_particles, iter, dt, alpha, batch_size)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class GCN_CBO:
  cdef CGCN_CBO *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double lambda_,
    double epsilon,
    double beta,
    double sigma,
    double alpha,
    double sigma_cn,
    bool independent_noise
  ):
    self.thisptr = new CGCN_CBO(bounds, n_particles, iter, dt, lambda_, epsilon, beta, sigma, alpha, sigma_cn, independent_noise)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class GCN_SBS:
  cdef CGCN_SBS *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double sigma,
    double sigma_cn,
  ):
    self.thisptr = new CGCN_SBS(bounds, n_particles, iter, dt, sigma, sigma_cn)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res
  
  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)

cdef class GCN_Langevin:
  cdef CGCN_Langevin *thisptr
  def __cinit__(
    self,
    bounds,
    int n_particles,
    int iter,
    double dt,
    double beta,
    double sigma_cn,
    bool independent_noise
  ):
    self.thisptr = new CGCN_Langevin(bounds, n_particles, iter, dt, beta, sigma_cn, independent_noise)

  def minimize(self, f):
    py_init()
    cdef PyObject* pyob_ptr = <PyObject*>f
    res = self.thisptr.py_minimize(pyob_ptr)
    return res

  def set_stop_criterion(self, stop_criterion):
    self.thisptr.set_stop_criterion(stop_criterion)