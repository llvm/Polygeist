//@ transform: tiling
// k strip-mined by 4.
void syrk__tiling(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int kk = 0; kk < m; kk += 4) {
      int ke = kk + 4 < m ? kk + 4 : m;
      for (int k = kk; k < ke; k++)
        for (int j = 0; j <= i; j++)
          C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
}
