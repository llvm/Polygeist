//@ routine: axpy
//@ kind: off_by_one
//@ expect: nomatch_or_equivalent
// y[i] += alpha*x[i+1] for i < n-1  (daxpy(n-1, alpha, x+1, y)).
void axpy__off_by_one(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n - 1; i++)
    y[i] += alpha * x[i + 1];
}
