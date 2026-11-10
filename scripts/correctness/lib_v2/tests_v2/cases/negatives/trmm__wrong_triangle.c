//@ routine: trmm
//@ kind: wrong_triangle
//@ expect: nomatch
// k < i: reads rows of B that this sweep has already overwritten.
void trmm__wrong_triangle(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < i; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
