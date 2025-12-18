/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/GCN/GCN.hh"
#include "optimizers/particles/common-noise/GCN/GCN_Langevin.hh"

void GCN_Langevin::set_stop_criterion(double stop_criterion)
{
  this->stop_criterion = stop_criterion;
  this->has_stop_criterion = true;
  this->base_opt.set_stop_criterion(stop_criterion);
}

result_eigen GCN_Langevin::minimize(function<double(dyn_vector)> f)
{
  GCN gcn(&this->base_opt, this->sigma, this->name);
  return gcn.minimize(f);
}