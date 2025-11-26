/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/common_noise.hh"
#include "optimizers/particles/common-noise/CN_SBS.hh"

void CN_SBS::set_stop_criterion(double stop_criterion)
{
  this->stop_criterion = stop_criterion;
  this->has_stop_criterion = true;
  this->base_opt.set_stop_criterion(stop_criterion);
}

result_eigen CN_SBS::minimize(function<double(dyn_vector)> f)
{
  Common_Noise cn(&this->base_opt, this->gamma, this->lambda, this->delta, static_cast<NoiseType>(this->moment), "CN_SBS");
  return cn.minimize(f);
}