/*
 * Created in 2024 by Gaëtan Serré
 */

#include "optimizers/decision/PolynomialFeatures.hh"

int comp(const int &n, const int &k)
{
  std::vector<int> aSolutions(k);
  aSolutions[0] = n - k + 1;

  for (int i = 1; i < k; ++i)
  {
    aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
  }

  return aSolutions[k - 1];
}

dyn_vector polynomial_features(dyn_vector &X, int degree)
{
  int n_features = X.size();
  int n_out = comp(n_features + degree, n_features) - 1;
  dyn_vector XP(n_out);

  copy(X.data(), X.data() + X.size(), XP.data());

  if (degree == 1)
    return XP;

  vector<int> indices(n_features + 1);
  for (int i = 0; i <= n_features; i++)
    indices[i] = i;

  int current_col = n_features;

  for (int i = 2; i <= degree; i++)
  {
    vector<int> new_indices(n_features + 1);
    int end = indices.back();
    for (int feature_idx = 0; feature_idx < n_features; feature_idx++)
    {
      int start = indices[feature_idx];
      new_indices[feature_idx] = current_col;
      int next_col = current_col + end - start;
      if (next_col <= current_col)
        break;
      for (int j = 0; j < end - start; j++)
      {
        XP(current_col + j) = X(feature_idx) * XP(start + j);
      }

      current_col = next_col;
    }
    new_indices[n_features] = current_col;
    indices = new_indices;
  }
  return XP;
}