//@ routine: gemm
//@ kind: read_before_final
//@ expect: nomatch
//@ square: true
// Uses C[i][k] (an output element still being accumulated) in place of A[i][k].
void gemm__read_before_final(int n, double alpha, double beta,
                             double C[n][n], double A[n][n], double B[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i][j] += alpha * C[i][k] * B[k][j];
  }
}
