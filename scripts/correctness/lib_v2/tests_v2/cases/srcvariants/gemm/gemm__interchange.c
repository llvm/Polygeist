//@ transform: interchange
// i-j-k order: the reduction loop innermost.
void gemm__interchange(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      C[i][j] *= beta;
      for (int k = 0; k < nk; k++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
}
