//@ routine: trmm
//@ kind: read_before_final
//@ expect: nomatch
// Rows visited bottom-up: B[k][j] (k > i) has already been overwritten.
void trmm__read_before_final(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = m - 1; i >= 0; i--)
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
