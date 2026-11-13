//@ transform: unroll2
void gemver__unroll2(int n, double alpha, double beta, double A[n][n], double u1[n], double v1[n], double u2[n], double v2[n], double w[n], double x[n], double y[n], double z[n]) {
  for (int i = 0; i < n; i++) {
    int j = 0;
    for (; j + 1 < n; j += 2) {
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      A[i][j + 1] = A[i][j + 1] + u1[i] * v1[j + 1] + u2[i] * v2[j + 1];
    }
    for (; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];
  for (int i = 0; i < n; i++)
    x[i] = x[i] + z[i];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];
}
