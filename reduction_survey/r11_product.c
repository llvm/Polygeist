// Product reduction (multiplicative monoid).
double prod(int n, double *x) {
  double p = 1;
  for (int i = 0; i < n; i++) p *= x[i];
  return p;
}
