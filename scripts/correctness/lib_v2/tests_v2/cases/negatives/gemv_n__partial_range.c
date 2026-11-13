//@ routine: gemv_n
//@ kind: partial_range
//@ expect: nomatch
// beta scaling over all m rows, accumulation only over rows 0..m-2.
void gemv_n__partial_range(int m, int n, double alpha, double beta,
                           double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++)
    y[i] = beta * y[i];
  for (int i = 0; i < m - 1; i++)
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * x[j];
}
