//@ transform: unroll2
// k unrolled by two with a remainder iteration.
void gemm__unroll2(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    int k = 0;
    for (; k + 1 < nk; k += 2)
      for (int j = 0; j < nj; j++) {
        C[i][j] += alpha * A[i][k] * B[k][j];
        C[i][j] += alpha * A[i][k + 1] * B[k + 1][j];
      }
    for (; k < nk; k++)
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
  }
}
