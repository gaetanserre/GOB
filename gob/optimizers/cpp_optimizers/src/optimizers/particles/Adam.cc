/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/particles/Adam.hh"

Eigen::MatrixXd Adam::step(Eigen::MatrixXd grads, Eigen::MatrixXd params)
{
  this->t++;
  this->state_m = this->beta1 * this->state_m + (1 - this->beta1) * grads;
  this->state_v = (this->beta2 * this->state_v).array() + (1 - this->beta2) * grads.array().pow(2);

  Eigen::MatrixXd m_hat = this->state_m.array() / (1 - pow(this->beta1, this->t));
  Eigen::MatrixXd v_hat = this->state_v.array() / (1 - pow(this->beta2, this->t));

  if (this->amsgrad)
  {
    this->state_v_max = this->state_v_max.cwiseMax(v_hat);
    return params.array() - this->lr * m_hat.array() / (this->state_v_max.array().sqrt() + this->epsilon);
  }
  Eigen::MatrixXd res = this->lr * m_hat.array() / (v_hat.array().sqrt() + this->epsilon);
  for (int i = 0; i < params.rows(); i++)
  {
    printf("Norm %d: %f\n", i, res.row(i).norm());
  }
  return params - res;
}