//@ routine: ger
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
// DGER has no beta; the analogous near miss drops alpha although it is a
// parameter (equals dger with alpha == 1).
void ger__missing_alpha(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += x[i] * y[j];
}
