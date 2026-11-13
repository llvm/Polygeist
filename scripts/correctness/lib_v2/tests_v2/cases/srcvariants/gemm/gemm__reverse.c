//@ transform: reverse
// Both parallel loops (i, j) run backwards; the k order is unchanged.
void gemm__reverse(int ni, int nj, int nk, double alpha, double beta, double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
  for (int i = ni - 1; i >= 0; i--) {
    for (int j = nj - 1; j >= 0; j--)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++)
      for (int j = nj - 1; j >= 0; j--)
        C[i][j] += alpha * A[i][k] * B[k][j];
  }
}
