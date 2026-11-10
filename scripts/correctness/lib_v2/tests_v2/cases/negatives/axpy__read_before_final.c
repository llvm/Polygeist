//@ routine: axpy
//@ kind: read_before_final
//@ expect: nomatch
// Reads y[n-1-i]: the original value for i < n/2, the updated one afterwards.
void axpy__read_before_final(int n, double alpha, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] += alpha * y[n - 1 - i];
}
