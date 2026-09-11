// Loop result has multiple uses.
double sum_and_sumsq_diff(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  return s + s * s;
}
