//@ kind: explicit
//@ expect: match_correct
//@ square: true
// gemm(A, A, C): both inputs are the same array (read-only aliasing; legal).
void gemm_AA(int n, double alpha, double beta, double C[n][n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      C[i][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        C[i][j] += alpha * A[i][k] * A[k][j];
  }
}
