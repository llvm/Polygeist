// Loop result combined with a runtime value in an arith expression, then returned.
double mean(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  return s / n;
}
