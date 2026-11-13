//@ transform: temp_scalar
void syrk__temp_scalar(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) {
      double s = beta * C[i][j];
      for (int k = 0; k < m; k++)
        s += alpha * A[i][k] * A[j][k];
      C[i][j] = s;
    }
}
