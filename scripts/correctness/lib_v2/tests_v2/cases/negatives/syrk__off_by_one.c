//@ routine: syrk
//@ kind: off_by_one
//@ expect: nomatch
// Accumulation j < i (diagonal skipped) while the scaling covers j <= i.
void syrk__off_by_one(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = 0; j < i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
  }
}
