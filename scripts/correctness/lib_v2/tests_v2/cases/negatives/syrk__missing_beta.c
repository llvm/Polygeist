//@ routine: syrk
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
void syrk__missing_beta(int n, int m, double alpha, double beta, double C[n][n], double A[n][m]) {
  for (int i = 0; i < n; i++)
    for (int k = 0; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
}
