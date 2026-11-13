//@ routine: gemm
//@ kind: wrong_index
//@ expect: nomatch
//@ square: true
// B[k][i] instead of B[k][j]: the update no longer depends on j.
void gemm__wrong_index(int n, double alpha, double beta,
                       double C[n][n], double A[n][n], double B[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i][j] += alpha * A[i][k] * B[k][i];
  }
}
