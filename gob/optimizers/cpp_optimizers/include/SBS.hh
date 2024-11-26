/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizer.hh"

class SBS : public Optimizer
{
public:
  SBS(
      vec_bounds bounds,
      int n_particles = 1,
      int svgd_iter = 4,
      std::vector<int> k_iter = {10000},
      double sigma = 1e-2,
      double lr = 0.5) : Optimizer(bounds, "SBS")
  {
    this->n_particles = n_particles;
    this->svgd_iter = svgd_iter;
    this->k_iter = k_iter;
    this->sigma = sigma;
    this->lr = lr;
  };

  virtual result_eigen minimize(function<double(dyn_vector x)> f);

  int n_particles;
  int svgd_iter;
  std::vector<int> k_iter;
  double sigma;
  double lr;

private:
  Eigen::MatrixXd rbf(Eigen::MatrixXd &X);
  Eigen::MatrixXd rbf_grad(Eigen::MatrixXd &X, Eigen::MatrixXd *rbf);
};