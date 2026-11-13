//@ routine: syrk
//@ kind: read_before_final
//@ expect: nomatch
//@ square: true
// Uses C[j][k] (output elements, row i still accumulating) instead of A[j][k].
void syrk__read_before_final(int n, double alpha, double beta, double C[n][n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * C[j][k];
  }
}
