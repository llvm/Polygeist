// Loop result passes through a pure scalar op (sqrt) before return.
double dnrm2(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return __builtin_sqrt(s);
}
