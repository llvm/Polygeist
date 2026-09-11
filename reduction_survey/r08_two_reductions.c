// Two iter_args in a single loop (sum + sum-of-squares).
void mean_sumsq(int n, double *x, double *m, double *q) {
  double s = 0, ss = 0;
  for (int i = 0; i < n; i++) {
    s += x[i];
    ss += x[i] * x[i];
  }
  *m = s;
  *q = ss;
}
