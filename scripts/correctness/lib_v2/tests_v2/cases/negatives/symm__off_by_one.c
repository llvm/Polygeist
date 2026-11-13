//@ routine: symm
//@ kind: off_by_one
//@ expect: nomatch
// k <= i: the diagonal is applied inside the loop as well.
void symm__off_by_one(int m, int n, double alpha, double beta,
                      double C[m][n], double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double temp2 = 0;
      for (int k = 0; k <= i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
}
