// Reduction used as a divisor in a subsequent loop.
void normalize(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  for (int i = 0; i < n; i++) x[i] /= s;
}
