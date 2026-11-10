//@ routine: gemv_t
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
// Only the first half of y is computed (a GEMV on a sub-block).
void gemv_t__partial_range(int m, int n, double alpha, double beta,
                           double A[m][n], double x[m], double y[n]) {
  for (int j = 0; j < n / 2; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n / 2; j++)
      y[j] += alpha * A[i][j] * x[i];
}
