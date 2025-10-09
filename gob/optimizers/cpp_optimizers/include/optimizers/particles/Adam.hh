/*
 * Created in 2024 by Gaëtan Serré
 */

#include "utils.hh"

class Adam
{
public:
  Adam(int n, int d, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double amsgrad = false)
  {
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->amsgrad = amsgrad;
    this->t = 0;
    this->state_m = Eigen::MatrixXd::Zero(n, d);
    this->state_v = Eigen::MatrixXd::Zero(n, d);
    this->state_v_max = Eigen::MatrixXd::Zero(n, d);
  }

  Eigen::MatrixXd step(Eigen::MatrixXd grads, Eigen::MatrixXd params);

private:
  double lr;
  double beta1;
  double beta2;
  double epsilon;
  bool amsgrad;
  Eigen::MatrixXd state_m;
  Eigen::MatrixXd state_v;
  Eigen::MatrixXd state_v_max;
  int t;
};