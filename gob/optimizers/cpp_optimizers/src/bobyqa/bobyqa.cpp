#include "impl.hpp"

#include <optimizers/decision/bobyqa.hh>

double bobyqa(
    const BobyqaFunction function,
    const long n,
    const long npt,
    double *x,
    const double *xl,
    const double *xu,
    const double rhobeg,
    const double rhoend,
    const long maxfun,
    double *w)
{
  return bobyqa_detail::impl([=](long n, const double *x) -> double
                             { return function(n, x); }, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, w);
}

double bobyqa_closure(
    BobyqaClosure *const closure,
    const long n,
    const long npt,
    double *x,
    const double *xl,
    const double *xu,
    const double rhobeg,
    const double rhoend,
    const long maxfun,
    double *w)
{
  return bobyqa_detail::impl([&](long n, const double *x) -> double
                             { return closure->function(closure->data, n, x); }, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, w);
}

double bobyqa_closure_const(
    const BobyqaClosureConst *const closure,
    const long n,
    const long npt,
    double *x,
    const double *xl,
    const double *xu,
    const double rhobeg,
    const double rhoend,
    const long maxfun,
    double *w)
{
  return bobyqa_detail::impl([&](long n, const double *x) -> double
                             { return closure->function(closure->data, n, x); }, n, npt, x, xl, xu, rhobeg, rhoend, maxfun, w);
}

template <class F>
BobyqaClosure make_closure(F &function)
{
  struct Wrap
  {
    static double call(void *data, long n, const double *values)
    {
      dyn_vector x = Eigen::Map<const dyn_vector>(values, n);
      return reinterpret_cast<F *>(data)->operator()(x);
    }
  };
  return BobyqaClosure{&function, &Wrap::call};
}

result_eigen run_bobyqa(
    const vec_bounds bounds,
    const dyn_vector x_dyn,
    const double radius,
    const int maxfun,
    function<double(dyn_vector x)> &f)
{
  auto closure = make_closure(f);
  const long variables_count = x_dyn.size();
  const long number_of_interpolation_conditions = variables_count + 2;
  vector<double> variables_values(variables_count);
  for (int i = 0; i < variables_count; i++)
  {
    variables_values[i] = x_dyn[i];
  }
  vector<double> lower_bound(variables_count);
  vector<double> upper_bound(variables_count);
  for (int i = 0; i < variables_count; i++)
  {
    lower_bound[i] = bounds[i][0];
    upper_bound[i] = bounds[i][1];
  }
  const double initial_trust_region_radius = radius;
  const double final_trust_region_radius = radius / 1e5;
  const size_t working_space_size = BOBYQA_WORKING_SPACE_SIZE(
      variables_count, number_of_interpolation_conditions);
  vector<double> working_space(working_space_size);
  const double result = bobyqa_closure(
      &closure,
      variables_count,
      number_of_interpolation_conditions,
      variables_values.data(),
      lower_bound.data(),
      upper_bound.data(),
      initial_trust_region_radius,
      final_trust_region_radius,
      maxfun,
      working_space.data());

  dyn_vector x(variables_count);
  for (int i = 0; i < variables_count; i++)
  {
    x[i] = variables_values[i];
  }

  return {x, result};
}