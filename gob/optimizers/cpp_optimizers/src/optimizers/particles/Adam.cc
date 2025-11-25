/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/Adam.hh"

void Adam::step(Eigen::MatrixXd *param, Eigen::MatrixXd grad, const int &time)
{
  if (time == 0)
  {
    this->m = Eigen::MatrixXd::Zero(grad.rows(), grad.cols());
    this->v = Eigen::MatrixXd::Zero(grad.rows(), grad.cols());
  }
  this->m = this->beta1 * this->m + (1 - this->beta1) * grad;
  this->v = (this->beta2 * this->v).array() + (1 - this->beta2) * grad.array().pow(2);

  Eigen::MatrixXd m_hat = this->m / (1 - pow(this->beta1, time + 1));
  Eigen::MatrixXd v_hat = this->v / (1 - pow(this->beta2, time + 1));

  Eigen::MatrixXd approx_grad = m_hat.array() / (v_hat.array().sqrt() + this->epsilon);
  (*param) = (*param) + this->dt * approx_grad;
  this->approx_dt = this->dt * (approx_grad.norm() / grad.norm());
}

double Adam::get_dt()
{
  return this->approx_dt;
}