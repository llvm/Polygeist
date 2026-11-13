//@ routine: ger
//@ kind: extra_term
//@ expect: nomatch
void ger__extra_term(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += alpha * x[i] * y[j] + x[i];
}
