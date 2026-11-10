//@ transform: reverse
void syrk__reverse(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = n - 1; i >= 0; i--) {
    for (int j = i; j >= 0; j--)
      C[i][j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = i; j >= 0; j--)
        C[i][j] += alpha * A[i][k] * A[j][k];
  }
}
