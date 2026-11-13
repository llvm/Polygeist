//@ routine: syr2k
//@ kind: wrong_index
//@ expect: nomatch
// First term uses B[j][k] instead of B[i][k].
void syr2k__wrong_index(int n, int m, double alpha, double beta,
                        double C[n][n], double A[n][m], double B[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += A[j][k] * alpha * B[j][k] + B[j][k] * alpha * A[i][k];
  }
}
