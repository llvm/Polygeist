// Max reduction via conditional update.
double maxabs(int n, double *x) {
  double m = 0;
  for (int i = 0; i < n; i++) {
    double a = x[i] < 0 ? -x[i] : x[i];
    if (a > m) m = a;
  }
  return m;
}
