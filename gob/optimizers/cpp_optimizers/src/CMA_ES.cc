/*
 * Created in 2024 by Gaëtan Serré
 */

#include "CMA_ES.hh"
#include "libcmaes/cmaes.h"
using namespace libcmaes;

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

  CMAParameters<> cmaparams(m0, this->sigma);
  cmaparams.set_max_iter(this->n_eval);

  if (this->has_stop_criteria)
  {
    cmaparams.set_ftarget(this->stop_criteria);
  }

  FitFunc f_ = [&f](const double *x, const int N)
  {
    dyn_vector xvec = dyn_vector::Map(x, N);
    return f(xvec);
  };
  auto opt = cmaes<>(f_, cmaparams);

  vector<Candidate> candidates = opt.candidates();
  Candidate best_candidate = candidates[0];

  return make_pair(best_candidate.get_x_dvec(), best_candidate.get_fvalue());
}