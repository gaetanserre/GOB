/*
 * Created in 2024 by Gaëtan Serré
 */

#include "AdaLIPO_P.hh"

bool AdaLIPO_P::Bernoulli(double p)
{
  std::bernoulli_distribution d(p);
  return d(this->re);
}

bool AdaLIPO_P::slope_stop_condition(std::deque<int> last_nb_samples)
{
  float slope = (last_nb_samples.back() - last_nb_samples.front()) / (this->window_size - 1);
  return slope > this->max_slope;
}

bool lipo_condition(
    dyn_vector x,
    std::vector<dyn_vector> samples,
    std::vector<double> values,
    double k_hat)
{
  double max_values = max_vec(values);
  std::vector<double> norms(samples.size());
  for (int i = 0; i < samples.size(); i++)
  {
    norms[i] = values[i] + k_hat * (x - samples[i]).norm();
  }
  return max_values <= min_vec(norms);
}

double AdaLIPO_P::minimize(function<double(dyn_vector x)> f)
{
  double alpha = 1e-2;
  double k_hat = 0;

  auto p = [](int t) -> double
  {
    if (t == 1)
      return 1;
    return 1 / log(t);
  };

  std::vector<double> ratios;

  std::vector<dyn_vector> samples;
  std::vector<double> values;

  samples.push_back(unif_random_vector(this->re, this->bounds));
  values.push_back(-f(samples.back()));

  int nb_samples = 1;
  std::deque<int> last_nb_samples(this->window_size, 0);
  last_nb_samples[0] = 1;

  for (int t = 1; t < this->n_eval; t++)
  {
    if (this->Bernoulli(p(t)))
    {
      dyn_vector x = unif_random_vector(this->re, this->bounds);
      nb_samples++;
      last_nb_samples[last_nb_samples.size() - 1] = nb_samples;
      samples.push_back(x);
      values.push_back(-f(x));
    }
    else
    {
      while (true)
      {
        dyn_vector x = unif_random_vector(this->re, this->bounds);
        nb_samples++;
        last_nb_samples[last_nb_samples.size() - 1] = nb_samples;
        if (lipo_condition(x, samples, values, k_hat))
        {
          samples.push_back(x);
          values.push_back(-f(x));
          break;
        }

        if (this->slope_stop_condition(last_nb_samples))
        {
          return -max_vec(values);
        }
      }
    }

    double value = values.back();
    dyn_vector x = samples.back();
    for (int i = 0; i < t; i++)
    {
      ratios.push_back(abs(value - values[i]) / (x - samples[i]).norm());
    }
    int i = ceil(log(max_vec(ratios)) / log(1 + alpha));
    k_hat = pow(1 + alpha, i);

    last_nb_samples.pop_front();
    last_nb_samples.push_back(0);
  }
  return -max_vec(values);
}