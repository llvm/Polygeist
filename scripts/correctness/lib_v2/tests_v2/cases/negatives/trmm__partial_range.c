//@ routine: trmm
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
void trmm__partial_range(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 1; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
