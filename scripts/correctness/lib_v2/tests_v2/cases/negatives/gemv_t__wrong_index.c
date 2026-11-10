//@ routine: gemv_t
//@ kind: wrong_index
//@ expect: nomatch
//@ square: true
// x[j] instead of x[i].
void gemv_t__wrong_index(int n, double alpha, double beta,
                         double A[n][n], double x[n], double y[n]) {
  for (int j = 0; j < n; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * x[j];
}
