/* Positive (literal binding): rank-1 update without alpha -> DGER, alpha = 1.0. */
void ger_unit(int m, int n, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] = A[i][j] + x[i] * y[j];
}
