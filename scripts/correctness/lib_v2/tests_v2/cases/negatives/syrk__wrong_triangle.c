//@ routine: syrk
//@ kind: wrong_triangle
//@ expect: nomatch
// Scales the lower triangle but accumulates into the upper one.
void syrk__wrong_triangle(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = i; j < n; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
  }
}
