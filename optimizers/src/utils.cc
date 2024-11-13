/*
 * Created in 2024 by Gaëtan Serré
 */

#include "utils.hh"

double max_vec(std::vector<double> &v)
{
  return *std::max_element(v.begin(), v.end());
}

double min_vec(std::vector<double> &v)
{
  return *std::min_element(v.begin(), v.end());
}

vec_bounds create_rect_bounds(double lb, double ub, int n)
{
  vec_bounds bounds(n, std::vector<double>(2));
  for (int i = 0; i < n; i++)
  {
    bounds[i][0] = lb;
    bounds[i][1] = ub;
  }
  return bounds;
}

double unif_random_double(std::default_random_engine &re, double lb, double ub)
{
  std::uniform_real_distribution<double> unif(lb, ub);
  double res = unif(re);
  re.seed(std::chrono::system_clock::now().time_since_epoch().count());
  return res;
}

dyn_vector unif_random_vector(std::default_random_engine &re, vec_bounds &bounds)
{
  int n = bounds.size();
  dyn_vector x(n);
  for (int i = 0; i < n; i++)
  {
    x(i) = unif_random_double(re, bounds[i][0], bounds[i][1]);
  }
  return x;
}

void print_vector(dyn_vector &x)
{
  std::cout << '[';
  for (int i = 0; i < x.size() - 1; i++)
  {
    std::cout << x(i) << ", ";
  }
  std::cout << x(x.size() - 1) << ']' << std::endl;
}