//@ kind: explicit
//@ expect: reject_or_correct
//@ square: true
// gemm(A, C, C): the output also plays the role of B.
void gemm_C_is_B(int n, double alpha, double beta, double C[n][n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i][j] += alpha * A[i][k] * C[k][j];
  }
}
