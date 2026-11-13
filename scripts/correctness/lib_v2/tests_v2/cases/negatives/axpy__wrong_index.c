//@ routine: axpy
//@ kind: wrong_index
//@ expect: nomatch
// x[0] instead of x[i].
void axpy__wrong_index(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] += alpha * x[0];
}
