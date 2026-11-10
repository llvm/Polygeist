//@ transform: tiling
// j strip-mined by 4.
void gemm__tiling(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int jj = 0; jj < nj; jj += 4) {
        int je = jj + 4 < nj ? jj + 4 : nj;
        for (int j = jj; j < je; j++)
          C[i][j] += alpha * A[i][k] * B[k][j];
      }
  }
}
