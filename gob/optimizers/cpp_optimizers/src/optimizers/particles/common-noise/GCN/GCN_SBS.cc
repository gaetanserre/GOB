/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/common-noise/GCN/GCN.hh"
#include "optimizers/particles/common-noise/GCN/GCN_SBS.hh"

void GCN_SBS::set_stop_criterion(double stop_criterion)
{
  this->stop_criterion = stop_criterion;
  this->has_stop_criterion = true;
  this->base_opt.set_stop_criterion(stop_criterion);
}

result_eigen GCN_SBS::minimize(function<double(dyn_vector)> f)
{
  GCN gcn(&this->base_opt, this->sigma, this->name, false);
  return gcn.minimize(f);
}