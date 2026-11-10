//@ transform: unroll2
void atax__unroll2(int m, int n, double A[m][n], double x[n], double y[n], double tmp[m]) {
  for (int i = 0; i < n; i++)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    tmp[i] = 0.0;
    int j = 0;
    for (; j + 1 < n; j += 2) {
      tmp[i] = tmp[i] + A[i][j] * x[j];
      tmp[i] = tmp[i] + A[i][j + 1] * x[j + 1];
    }
    for (; j < n; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < n; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
