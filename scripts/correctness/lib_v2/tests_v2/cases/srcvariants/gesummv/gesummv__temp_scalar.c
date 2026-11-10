//@ transform: temp_scalar
void gesummv__temp_scalar(int n, double alpha, double beta, double A[n][n], double B[n][n], double tmp[n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    double t = 0.0, u = 0.0;
    for (int j = 0; j < n; j++) {
      t = A[i][j] * x[j] + t;
      u = B[i][j] * x[j] + u;
    }
    tmp[i] = t;
    y[i] = alpha * t + beta * u;
  }
}
