//@ transform: flatten_1d
//@ extents: x1=n, x2=n, y_1=n, y_2=n, A=n*n
void mvt__flatten_1d(int n, double *x1, double *x2, double *y_1, double *y_2, double *A) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x1[i] = x1[i] + A[i * n + j] * y_1[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x2[i] = x2[i] + A[j * n + i] * y_2[j];
}
