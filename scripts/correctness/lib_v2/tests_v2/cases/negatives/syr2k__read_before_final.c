//@ routine: syr2k
//@ kind: read_before_final
//@ expect: nomatch
//@ square: true
// Second term reads C[i][k] (output) instead of A[i][k].
void syr2k__read_before_final(int n, double alpha, double beta,
                              double C[n][n], double A[n][n], double B[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * C[i][k];
  }
}
