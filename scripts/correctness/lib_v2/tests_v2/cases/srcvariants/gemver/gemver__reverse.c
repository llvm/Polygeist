//@ transform: reverse
void gemver__reverse(int n, double alpha, double beta, double A[n][n], double u1[n], double v1[n], double u2[n], double v2[n], double w[n], double x[n], double y[n], double z[n]) {
  for (int i = n - 1; i >= 0; i--)
    for (int j = n - 1; j >= 0; j--)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  for (int i = n - 1; i >= 0; i--)
    for (int j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];
  for (int i = n - 1; i >= 0; i--)
    x[i] = x[i] + z[i];
  for (int i = n - 1; i >= 0; i--)
    for (int j = 0; j < n; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];
}
