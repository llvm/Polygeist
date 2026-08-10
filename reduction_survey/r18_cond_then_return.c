// Reduction used in a branch condition.
int any_negative(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  if (s < 0) return 1;
  return 0;
}
