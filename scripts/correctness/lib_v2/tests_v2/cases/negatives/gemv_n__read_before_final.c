//@ routine: gemv_n
//@ kind: read_before_final
//@ expect: nomatch
//@ square: true
// Multiplies by y[j] (output, partly updated) instead of x[j].
void gemv_n__read_before_final(int n, double alpha, double beta,
                               double A[n][n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * y[j];
  }
}
