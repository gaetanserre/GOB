/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

enum NoiseType
{
  M1 = 0,
  M2 = 1,
  VAR = 2,
  MVAR = 3
};

struct common_dynamic
{
  dyn_vector drift;
  Eigen::MatrixXd noise;
};

class GCN : public Optimizer
{
public:
  GCN(
      Particles_Optimizer *base_optimizer,
      double sigma,
      std::string name) : Optimizer(base_optimizer->bounds, name)
  {
    this->base_opt = base_optimizer;
    this->sigma = sigma;
  }

  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  Particles_Optimizer *base_opt;
  double sigma;

  Eigen::MatrixXd compute_noise(const Eigen::MatrixXd &particles);
  void update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples, const int &time_);
};