// Argmax: track value AND index together (two iter_args of different types).
int idamax(int n, double *x) {
  double m = x[0];
  int k = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > m) { m = x[i]; k = i; }
  }
  return k;
}
