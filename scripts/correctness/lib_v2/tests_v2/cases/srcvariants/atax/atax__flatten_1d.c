//@ transform: flatten_1d
//@ extents: A=m*n, x=n, y=n, tmp=m
void atax__flatten_1d(int m, int n, double *A, double *x, double *y, double *tmp) {
  for (int i = 0; i < n; i++)
    y[i] = 0;
  for (int i = 0; i < m; i++) {
    tmp[i] = 0.0;
    for (int j = 0; j < n; j++)
      tmp[i] = tmp[i] + A[i * n + j] * x[j];
    for (int j = 0; j < n; j++)
      y[j] = y[j] + A[i * n + j] * tmp[i];
  }
}
