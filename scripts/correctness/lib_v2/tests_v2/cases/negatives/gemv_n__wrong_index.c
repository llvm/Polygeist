//@ routine: gemv_n
//@ kind: wrong_index
//@ expect: nomatch
//@ square: true
// x[i] instead of x[j]: y[i] += alpha * x[i] * rowsum(A[i]).
void gemv_n__wrong_index(int n, double alpha, double beta,
                         double A[n][n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * x[i];
  }
}
