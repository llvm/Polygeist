//@ transform: split_reduction
void gesummv__split_reduction(int n, double alpha, double beta, double A[n][n], double B[n][n], double tmp[n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    double t0 = 0.0, t1 = 0.0, u0 = 0.0, u1 = 0.0;
    for (int j = 0; j < n / 2; j++) {
      t0 = A[i][j] * x[j] + t0;
      u0 = B[i][j] * x[j] + u0;
    }
    for (int j = n / 2; j < n; j++) {
      t1 = A[i][j] * x[j] + t1;
      u1 = B[i][j] * x[j] + u1;
    }
    tmp[i] = t0 + t1;
    y[i] = alpha * tmp[i] + beta * (u0 + u1);
  }
}
