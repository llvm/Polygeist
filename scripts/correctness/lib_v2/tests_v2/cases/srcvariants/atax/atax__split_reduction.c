//@ transform: split_reduction
void atax__split_reduction(int m, int n, double A[m][n], double x[n], double y[n], double tmp[m]) {
  for (int i = 0; i < n; i++)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    double s0 = 0.0, s1 = 0.0;
    for (int j = 0; j < n / 2; j++)
      s0 += A[i][j] * x[j];
    for (int j = n / 2; j < n; j++)
      s1 += A[i][j] * x[j];
    tmp[i] = s0 + s1;
    for (int j = 0; j < n; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
