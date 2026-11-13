//@ routine: gemv_n
//@ kind: positive_control
//@ expect: match
// Row-major y := alpha*A*x + beta*y  (A is m x n).
void gemv_n__pos(int m, int n, double alpha, double beta,
                 double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * x[j];
  }
}
