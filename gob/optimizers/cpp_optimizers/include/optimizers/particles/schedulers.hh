/*
 * Created in 2025 by Gaëtan Serré
 */

class Scheduler
{
public:
  Scheduler() = default;

  virtual ~Scheduler() = default;

  virtual void step() {}
};

class LinearScheduler : public Scheduler
{
public:
  LinearScheduler(double *param, double coeff)
      : param(param), coeff(coeff) {}

  void step() override
  {
    *this->param *= this->coeff;
  }

private:
  double *param;
  double coeff;
};