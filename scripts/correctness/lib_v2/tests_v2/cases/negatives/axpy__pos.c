//@ routine: axpy
//@ kind: positive_control
//@ expect: match
// y := alpha*x + y.
void axpy__pos(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] += alpha * x[i];
}
