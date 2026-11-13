//@ routine: ger
//@ kind: read_before_final
//@ expect: nomatch
// Uses A[0][j] (row 0 is updated first) instead of y[j].
void ger__read_before_final(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += alpha * x[i] * A[0][j];
}
