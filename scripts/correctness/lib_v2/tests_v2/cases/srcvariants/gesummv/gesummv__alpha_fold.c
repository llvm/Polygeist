//@ transform: alpha_fold
// alpha / beta applied inside the reductions instead of at the end.
void gesummv__alpha_fold(int n, double alpha, double beta, double A[n][n], double B[n][n], double tmp[n], double x[n], double y[n]) {
  for (int i = 0; i < n; i++) {
    tmp[i] = 0.0;
    y[i] = 0.0;
    for (int j = 0; j < n; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = (beta * B[i][j]) * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + y[i];
  }
}
