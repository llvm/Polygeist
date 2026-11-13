//@ routine: gemv_n
//@ kind: wrong_index
//@ expect: nomatch_or_equivalent
//@ square: true
// A[j][i] instead of A[i][j]: a genuine GEMV with the transposed matrix.
void gemv_n__wrong_index_transposed(int n, double alpha, double beta,
                                    double A[n][n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[j][i] * x[j];
  }
}
