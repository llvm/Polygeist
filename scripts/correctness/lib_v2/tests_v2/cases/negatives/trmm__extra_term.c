//@ routine: trmm
//@ kind: extra_term
//@ expect: nomatch
void trmm__extra_term(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j] + A[k][i];
      B[i][j] = alpha * B[i][j];
    }
}
