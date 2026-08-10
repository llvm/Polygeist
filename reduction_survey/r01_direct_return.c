// Loop result returned directly.
double ddot(int n, double *x, double *y) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i] * y[i];
  return s;
}
