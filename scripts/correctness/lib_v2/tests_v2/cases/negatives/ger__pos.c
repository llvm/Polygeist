//@ routine: ger
//@ kind: positive_control
//@ expect: match
// Row-major rank-1 update A := alpha*x*y^T + A.
void ger__pos(int m, int n, double alpha, double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i][j] += alpha * x[i] * y[j];
}
