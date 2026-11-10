//@ kind: explicit
//@ expect: reject_or_correct
//@ square: true
// gemm(C, B, C): the output also plays the role of A (in-place product).
void gemm_C_is_A(int n, double alpha, double beta, double C[n][n], double B[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i][j] += alpha * C[i][k] * B[k][j];
  }
}
