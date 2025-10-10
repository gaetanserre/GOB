/*
 * Created in 2025 by Gaëtan Serré
 */

class Scheduler
{
public:
  Scheduler() = default;

  virtual ~Scheduler() = default;

  virtual void step() {}

  virtual void reset() {}
};

class LinearScheduler : public Scheduler
{
public:
  LinearScheduler(double *param, double coeff)
      : param(param), old_param(*param), coeff(coeff) {}

  void step() override
  {
    *this->param *= this->coeff;
  }

  void reset() override
  {
    *this->param = this->old_param;
  }

private:
  double *param;
  double old_param;
  double coeff;
};