//@ transform: temp_scalar
void atax__temp_scalar(int m, int n, double A[m][n], double x[n], double y[n], double tmp[m]) {
  for (int i = 0; i < n; i++)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    double t = 0.0;
    for (int j = 0; j < n; j++)
      t += A[i][j] * x[j];
    tmp[i] = t;
    for (int j = 0; j < n; j++)
      y[j] = y[j] + A[i][j] * t;
  }
}
