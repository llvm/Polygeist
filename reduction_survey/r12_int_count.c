// Integer counter incremented conditionally.
int count_positive(int n, double *x) {
  int c = 0;
  for (int i = 0; i < n; i++) if (x[i] > 0) c++;
  return c;
}
