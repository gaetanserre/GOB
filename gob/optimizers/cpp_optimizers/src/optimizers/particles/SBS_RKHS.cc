/*
 * Created in 2025 by Gaëtan Serré
 */

#include "optimizers/particles/SBS_RKHS.hh"
#include "optimizers/particles/noise.hh"
#include "optimizers/particles/particles_utils.hh"
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Dense>

Eigen::MatrixXd SBS_RKHS::rbf_grad(const Eigen::MatrixXd &particles, Eigen::MatrixXd *rbf_matrix)
{
  *rbf_matrix = rbf(particles, this->eval_sigma());
  Eigen::MatrixXd dxkxy = (particles.array().colwise() * rbf_matrix->colwise().sum().transpose().array()) - (*rbf_matrix * particles).array();
  return dxkxy;
}

Eigen::MatrixXd SBS_RKHS::compute_noise(const Eigen::MatrixXd &particles, const Eigen::MatrixXd &rbf_matrix)
{
  int d = particles.cols();
  Eigen::MatrixXd K_tmp = rbf_matrix.llt().solve(Eigen::MatrixXd::Identity(rbf_matrix.rows(), rbf_matrix.cols()));
  Eigen::MatrixXd K_inv = Eigen::kroneckerProduct(K_tmp, Eigen::MatrixXd::Identity(d, d)) / particles.rows();
  dyn_vector alphas_tmp = normal_random_variable(K_inv, &this->re)();
  Eigen::MatrixXd alphas = Eigen::Map<Eigen::MatrixXd>(alphas_tmp.data(), particles.rows(), d);
  return rbf_matrix * alphas;
}

double SBS_RKHS::eval_sigma()
{
  if (this->sigma == nullptr)
  {
    throw runtime_error("Sigma function is not defined.");
  }
  PyObject *args = PyTuple_New(0);
  PyObject *result = PyObject_CallObject(this->sigma, args);
  double sigma_value = 0.0;
  if (result)
  {
    sigma_value = PyFloat_AsDouble(result);
    Py_DECREF(result);
  }
  else
  {
    PyErr_Print();
  }
  Py_DECREF(args);
  return sigma_value;
}

dynamic SBS_RKHS::compute_dynamics(const Eigen::MatrixXd &particles, const function<double(dyn_vector x)> &f, vector<double> *evals)
{

  Eigen::MatrixXd grads(particles.rows(), this->bounds.size());
  for (int j = 0; j < particles.rows(); j++)
  {
    double f_x;
    grads.row(j) = -this->k * gradient(particles.row(j), f, &f_x);
    (*evals)[j] = f_x;
  }
  Eigen::MatrixXd kernel;
  Eigen::MatrixXd kernel_grad = this->rbf_grad(particles, &kernel);
  Eigen::MatrixXd noise = this->compute_noise(particles, kernel);

  for (int i = 0; i < particles.rows(); i++)
  {
    double eval = f(particles.row(i));
    (*evals)[i] = eval;
  }
  dyn_vector stddev = Eigen::VectorXd::Ones(particles.rows()) * pow(sqrt(2 * this->dt), this->theta - 1);
  return {((kernel * grads + kernel_grad) / particles.rows()), stddev, noise};
}