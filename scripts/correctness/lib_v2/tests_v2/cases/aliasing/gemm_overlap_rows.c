//@ kind: explicit
//@ expect: reject_or_correct
// Overlapping slices of one array X[n+1][n]: C = X[1:n+1], A = X[0:n].
void gemm_overlap_rows(int n, double alpha, double beta, double X[n + 1][n], double B[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      X[i + 1][j] *= beta;
    for (int k = 0; k < n; k++)
      for (int j = 0; j < n; j++)
        X[i + 1][j] += alpha * X[i][k] * B[k][j];
  }
}
