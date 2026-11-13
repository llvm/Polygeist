//@ routine: axpy
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
// DAXPY has no beta; the analogous near miss drops alpha although it is a parameter.
void axpy__missing_alpha(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] += x[i];
}
