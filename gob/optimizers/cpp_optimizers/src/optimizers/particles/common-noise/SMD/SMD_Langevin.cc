/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/SMD/SMD.hh"
#include "optimizers/particles/common-noise/SMD/SMD_Langevin.hh"

void SMD_Langevin::set_stop_criterion(double stop_criterion)
{
  this->stop_criterion = stop_criterion;
  this->has_stop_criterion = true;
  this->base_opt.set_stop_criterion(stop_criterion);
}

result_eigen SMD_Langevin::minimize(function<double(dyn_vector)> f)
{
  SMD smd(&this->base_opt, this->gamma, this->lambda, this->delta, static_cast<NoiseType>(this->moment), this->name);
  return smd.minimize(f);
}