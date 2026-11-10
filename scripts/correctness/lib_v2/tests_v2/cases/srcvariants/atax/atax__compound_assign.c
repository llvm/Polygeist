//@ transform: compound_assign
void atax__compound_assign(int m, int n, double A[m][n], double x[n], double y[n], double tmp[m]) {
  for (int i = 0; i < n; i++)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    tmp[i] = 0.0;
    for (int j = 0; j < n; j++)
      tmp[i] += A[i][j] * x[j];
    for (int j = 0; j < n; j++)
      y[j] += A[i][j] * tmp[i];
  }
}
