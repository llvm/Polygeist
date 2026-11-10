//@ routine: trmm
//@ kind: off_by_one
//@ expect: nomatch
// k starts at i: the diagonal term A[i][i]*B[i][j] is added to an implicit unit diagonal.
void trmm__off_by_one(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = i; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
