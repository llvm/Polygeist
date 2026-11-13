//@ routine: ger
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
// Rows 1..m-1 only.
void ger__partial_range(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 1; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += alpha * x[i] * y[j];
}
