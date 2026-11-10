//@ routine: trmm
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
// DTRMM has no beta; the analogous near miss drops alpha although it is a parameter.
void trmm__missing_alpha(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
}
