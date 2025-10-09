/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/CMA_ES.hh"
#include "libcmaes/cmaes.h"
using namespace libcmaes;

void CMA_ES::transform_bounds(vec_bounds bounds)
{
  const int dim = this->bounds.size();
  this->lbounds = new double[dim];
  this->ubounds = new double[dim];
  for (int i = 0; i < dim; i++)
  {
    this->lbounds[i] = this->bounds[i][0];
    this->ubounds[i] = this->bounds[i][1];
  }
}

result_eigen CMA_ES::minimize(function<double(dyn_vector x)> f)
{
  if (this->m0.size() == 0)
  {
    int n = this->bounds.size();
    this->m0 = vector<double>(n);
    for (int i = 0; i < n; i++)
    {
      this->m0[i] = unif_random_double(re, bounds[i][0], bounds[i][1]);
    }
  }

  GenoPheno<pwqBoundStrategy> gp(this->lbounds, this->ubounds, this->m0.size());
  CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(this->m0, this->sigma, -1, 0, gp);
  if (this->has_stop_criterion)
  {
    cmaparams.set_ftarget(this->stop_criterion);
  }
  cmaparams.set_max_fevals(this->n_eval);

  FitFunc f_ = [&f](const double *x, const int N)
  {
    dyn_vector xvec = dyn_vector::Map(x, N);
    return f(xvec);
  };

  CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy>>(f_, cmaparams);
  dyn_vector best_x = gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());
  double best_f = cmasols.get_best_seen_candidate().get_fvalue();

  return {best_x, best_f};
}