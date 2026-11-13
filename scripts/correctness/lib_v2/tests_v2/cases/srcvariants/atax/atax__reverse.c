//@ transform: reverse
// The y initialisation and the j loops run backwards.
void atax__reverse(int m, int n, double A[m][n], double x[n], double y[n], double tmp[m]) {
  for (int i = n - 1; i >= 0; i--)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    tmp[i] = 0.0;
    for (int j = 0; j < n; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (int j = n - 1; j >= 0; j--)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
