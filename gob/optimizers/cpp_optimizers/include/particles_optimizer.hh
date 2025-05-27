/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizer.hh"

class Particles_Optimizer : public Optimizer
{
public:
  SBS(
      vec_bounds bounds,
      function<function<double(dyn_vector x), Eigen::MatrixXd>> dynamics,
      int n_particles = 1,
      int iter = 100,
      double lr = 0.5) : Optimizer(bounds, "Particles_Optimizer")
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