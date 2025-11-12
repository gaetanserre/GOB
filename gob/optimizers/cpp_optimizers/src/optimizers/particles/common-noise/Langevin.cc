/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/Langevin.hh"
#include "optimizers/particles/common-noise/common_noise.hh"

void CN_Langevin::set_stop_criterion(double stop_criterion)
{
  this->stop_criterion = stop_criterion;
  this->has_stop_criterion = true;
  this->base_opt.set_stop_criterion(stop_criterion);
}

result_eigen CN_Langevin::minimize(function<double(dyn_vector x)> f)
{
  Common_Noise cn(&this->base_opt, 1.0, NoiseType::M1, "CN_Langevin");
  return cn.minimize(f);
}