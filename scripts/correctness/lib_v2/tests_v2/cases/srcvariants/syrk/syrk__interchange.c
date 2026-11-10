//@ transform: interchange
// i-j-k order: reduction innermost.
void syrk__interchange(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) {
      C[i][j] *= beta;
      for (int k = 0; k < m; k++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
}
