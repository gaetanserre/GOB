/*
 * Created in 2025 by Gaëtan Serré
 */

#include "trust_regions.hh"
#include "bobyqa.hh"

bool check_in_ball(vector<dyn_vector> centers, dyn_vector x, double radius)
{
  for (dyn_vector center : centers)
  {
    if ((center - x).norm() < radius)
    {
      return true;
    }
  }
  return false;
}

result_eigen TrustRegions::minimize(function<double(dyn_vector x)> f)
{
  vector<dyn_vector> centers;
  vector<pair<dyn_vector, double>> samples;

  auto compare_pair = [](pair<dyn_vector, double> a, pair<dyn_vector, double> b) -> bool
  {
    return a.second < b.second;
  };

  for (int i = 0; i < this->n_eval; i++)
  {
    int count = 0;
    while (true)
    {
      dyn_vector x = unif_random_vector(this->re, this->bounds);
      count++;
      if (
          !check_in_ball(centers, x, this->region_radius) &&
          (*this->decision)(samples, x, this->data, this->functions))
      {
        centers.push_back(x);
        result_eigen bobyqa_res = run_bobyqa(
            this->bounds,
            x,
            this->region_radius,
            this->bobyqa_eval,
            f);
        samples.push_back(make_pair(bobyqa_res.first, -bobyqa_res.second));
        sort(samples.begin(), samples.end(), compare_pair);
        break;
      }

      if (count >= this->max_samples)
      {
        result_eigen best = samples.back();
        return make_pair(best.first, -best.second);
      }
    }

    result_eigen best = samples.back();
    if (this->has_stop_criteria && -best.second <= this->stop_criteria)
    {
      return make_pair(best.first, -best.second);
    }

    if (this->callback != nullptr)
    {
      (*this->callback)(samples, this->data, this->functions);
    }
  }

  result_eigen best = samples.back();
  return make_pair(best.first, -best.second);
}