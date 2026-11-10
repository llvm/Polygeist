//@ routine: syrk
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
// Reduction over k = 1..m-1 only (a SYRK on A[:,1:]).
void syrk__partial_range(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 1; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
  }
}
