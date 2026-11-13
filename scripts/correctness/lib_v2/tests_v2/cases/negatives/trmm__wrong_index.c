//@ routine: trmm
//@ kind: wrong_index
//@ expect: nomatch
// A[k][k] instead of A[k][i].
void trmm__wrong_index(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][k] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
