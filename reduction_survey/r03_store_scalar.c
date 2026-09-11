// Loop result stored into a scalar memref (current path that works).
void asum_out(int n, double *x, double *out) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  *out = s;
}
