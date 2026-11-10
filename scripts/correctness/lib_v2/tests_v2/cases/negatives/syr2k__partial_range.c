//@ routine: syr2k
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
void syr2k__partial_range(int n, int m, double alpha, double beta,
                          double C[n][n], double A[n][m], double B[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 1; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
  }
}
