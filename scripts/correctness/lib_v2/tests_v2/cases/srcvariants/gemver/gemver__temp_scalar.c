//@ transform: temp_scalar
void gemver__temp_scalar(int n, double alpha, double beta, double A[n][n], double u1[n], double v1[n], double u2[n], double v2[n], double w[n], double x[n], double y[n], double z[n]) {
  for (int i = 0; i < n; i++) {
    double a = u1[i], b = u2[i];
    for (int j = 0; j < n; j++)
      A[i][j] = A[i][j] + a * v1[j] + b * v2[j];
  }
  for (int i = 0; i < n; i++) {
    double t = x[i];
    for (int j = 0; j < n; j++)
      t = t + beta * A[j][i] * y[j];
    x[i] = t;
  }
  for (int i = 0; i < n; i++)
    x[i] = x[i] + z[i];
  for (int i = 0; i < n; i++) {
    double t = w[i];
    for (int j = 0; j < n; j++)
      t = t + alpha * A[i][j] * x[j];
    w[i] = t;
  }
}
