/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/particles_optimizer.hh"

enum NoiseType
{
  M1 = 0,
  M2 = 1
};

class Common_Noise : public Optimizer
{
public:
  Common_Noise(
      Particles_Optimizer *base_optimizer,
      double gamma,
      NoiseType noise_type,
      std::string name) : Optimizer(base_optimizer->bounds, name)
  {
    this->base_opt = base_optimizer;
    this->gamma = gamma;
    this->noise_type = noise_type;
  }

  virtual result_eigen minimize(function<double(dyn_vector x)> f);
  virtual dynamic m1_dynamic(const Eigen::MatrixXd &particles);
  virtual dynamic m2_dynamic(const Eigen::MatrixXd &particles);

private:
  Particles_Optimizer *base_opt;
  double gamma;
  NoiseType noise_type;
  void update_particles(Eigen::MatrixXd *particles, function<double(dyn_vector x)> f, vector<double> *all_evals, vector<dyn_vector> *samples);
};