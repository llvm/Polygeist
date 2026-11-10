//@ transform: tiling
void gesummv__tiling(int n, double alpha, double beta, double A[n][n], double B[n][n], double tmp[n], double x[n], double y[n]) {
  for (int ii = 0; ii < n; ii += 4) {
    int ie = ii + 4 < n ? ii + 4 : n;
    for (int i = ii; i < ie; i++) {
      tmp[i] = 0.0;
      y[i] = 0.0;
      for (int j = 0; j < n; j++) {
        tmp[i] = A[i][j] * x[j] + tmp[i];
        y[i] = B[i][j] * x[j] + y[i];
      }
      y[i] = alpha * tmp[i] + beta * y[i];
    }
  }
}
