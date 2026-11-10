//@ kind: explicit
//@ expect: reject_or_correct
//@ square: true
// gemv(A, y, y): x aliases the output y.
void gemv_x_is_y(int n, double alpha, double beta, double A[n][n], double y[n]) {
  for (int i = 0; i < n; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * y[j];
  }
}
