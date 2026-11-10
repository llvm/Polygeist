//@ routine: gemv_t
//@ kind: read_before_final
//@ expect: nomatch
//@ square: true
// Multiplies by y[i] (output, partly accumulated) instead of x[i].
void gemv_t__read_before_final(int n, double alpha, double beta,
                               double A[n][n], double x[n], double y[n]) {
  for (int j = 0; j < n; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * y[i];
}
