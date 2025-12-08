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

class Common_Noise : public Optimizer
{
public:
  Common_Noise(
      Particles_Optimizer *base_optimizer,
      double gamma,
      double lambda,
      double delta,
      NoiseType noise_type,
      std::string name) : Optimizer(base_optimizer->bounds, name)
  {
    this->base_opt = base_optimizer;
    this->gamma = gamma;
    this->lambda = lambda;
    this->delta = delta;
    this->noise_type = noise_type;
  }

  virtual result_eigen minimize(function<double(dyn_vector)> f);

private:
  Particles_Optimizer *base_opt;
  double gamma;
  double lambda;
  double delta;
  NoiseType noise_type;

  common_dynamic m1_dynamic(const Eigen::MatrixXd &particles, const int &idx);
  common_dynamic square_dynamic(const Eigen::MatrixXd &particles, const int &idx, auto func);
  common_dynamic m2_dynamic(const Eigen::MatrixXd &particles, const int &idx);
  common_dynamic var_dynamic(const Eigen::MatrixXd &particles, const int &idx);
  common_dynamic mean_var_dynamic(const Eigen::MatrixXd &particles, const int &idx);

  void update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples, int &t);
};