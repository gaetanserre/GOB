/*
 * Created in 2025 by Gaëtan Serré
 */

#pragma once

#include "utils.hh"

class Scheduler
{
public:
  Scheduler() = default;

  virtual ~Scheduler() = default;

  virtual void step(Eigen::MatrixXd *param, Eigen::MatrixXd grad, const int &time) { printf("BAse sched\n"); }

  virtual void reset() {}

  virtual double get_dt() { return 0.0; }
};

class LinearScheduler : public Scheduler
{
public:
  LinearScheduler(const double dt, const double coeff)
      : first_dt(dt), coeff(coeff) {}

  void step(Eigen::MatrixXd *param, Eigen::MatrixXd grad, const int &time) override
  {
    this->dt = this->first_dt * pow(this->coeff, time + 1);
    *param += this->dt * grad;
  }

  double get_dt() override
  {
    return this->dt;
  }

  void reset() override
  {
    this->dt = this->first_dt;
  }

private:
  const double coeff;
  double first_dt;
  double dt = 0.0;
};