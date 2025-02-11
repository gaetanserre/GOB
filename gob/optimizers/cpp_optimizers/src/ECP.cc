/*
 * Created in 2025 by Gaëtan Serré
 */

#include "ECP.hh"

bool decision(
    const vector<dyn_vector> &points,
    const vector<double> &values,
    const dyn_vector &x,
    const double &epsilon,
    const function<double(dyn_vector x)> &f)
{
  double max_values = max_vec(values);
  vector<double> norms(points.size());
  for (int i = 0; i < points.size(); i++)
  {
    norms[i] = values[i] + epsilon * (x - points[i]).norm();
  }
  return max_values <= min_vec(norms);
}

result_eigen ECP::minimize(function<double(dyn_vector x)> f)
{
  vector<dyn_vector> points;
  vector<double> values;
  dyn_vector x = unif_random_vector(this->re, this->bounds);
  points.push_back(x);
  values.push_back(-f(x));
  int t = 1, h1 = 1, h2 = 0;
  while (t < this->n_eval)
  {
    x = unif_random_vector(this->re, this->bounds);
    h2++;
    if ((h2 - h1) > this->C)
    {
      this->epsilon = this->epsilon * this->theta;
      h2 = 0;
    }
    if (decision(points, values, x, this->epsilon, f))
    {
      points.push_back(x);
      values.push_back(-f(x));
      t++;
      h1 = h2;
      this->epsilon = this->epsilon * this->theta;
      h2 = 0;

      if (this->has_stop_criterion && -values.back() <= this->stop_criterion)
      {
        return {x, -values.back()};
      }
    }
  }

  int best_idx = argmax_vec(values);

  return {points[best_idx], -values[best_idx]};
}