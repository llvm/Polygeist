//@ routine: axpy
//@ kind: extra_term
//@ expect: nomatch
void axpy__extra_term(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] += alpha * x[i] + x[i] * x[i];
}
