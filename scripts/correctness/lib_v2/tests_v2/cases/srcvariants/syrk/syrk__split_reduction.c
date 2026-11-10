//@ transform: split_reduction
void syrk__split_reduction(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) {
      double s0 = 0.0, s1 = 0.0;
      for (int k = 0; k < m / 2; k++)
        s0 += A[i][k] * A[j][k];
      for (int k = m / 2; k < m; k++)
        s1 += A[i][k] * A[j][k];
      C[i][j] = beta * C[i][j] + alpha * (s0 + s1);
    }
}
