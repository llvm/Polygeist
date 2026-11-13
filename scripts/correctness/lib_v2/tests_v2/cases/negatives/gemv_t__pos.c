//@ routine: gemv_t
//@ kind: positive_control
//@ expect: match
// Row-major y := alpha*A^T*x + beta*y  (A is m x n, x has m, y has n).
void gemv_t__pos(int m, int n, double alpha, double beta,
                 double A[m][n], double x[m], double y[n]) {
  for (int j = 0; j < n; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * x[i];
}
