/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/schedulers.hh"

#pragma once

class Adam : public Scheduler
{
public:
  Adam(const double &dt, const double beta1 = 0.9, const double beta2 = 0.999) : beta1(beta1), beta2(beta2), dt(dt) {}

  void step(Eigen::MatrixXd *param, Eigen::MatrixXd grad, const int &time) override;
  double get_dt() override;

private:
  const double beta1;
  const double beta2;
  const double dt;

  Eigen::MatrixXd m;
  Eigen::MatrixXd v;
  double epsilon = 1e-8;
  double approx_dt = 0.0;
};