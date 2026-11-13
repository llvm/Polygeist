//@ routine: axpy
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
void axpy__partial_range(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n / 2; i++)
    y[i] += alpha * x[i];
}
