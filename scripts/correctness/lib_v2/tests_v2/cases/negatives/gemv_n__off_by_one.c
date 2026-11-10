//@ routine: gemv_n
//@ kind: off_by_one
//@ expect: nomatch_or_equivalent
// Reduction starts at j = 1 (skips column 0).
void gemv_n__off_by_one(int m, int n, double alpha, double beta,
                        double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++) {
    y[i] = beta * y[i];
    for (int j = 1; j < n; j++)
      y[i] += alpha * A[i][j] * x[j];
  }
}
