//@ routine: symm
//@ kind: partial_range
//@ expect: nomatch_or_equivalent
// Columns 1..n-1 only (a SYMM on the sub-block B[:,1:], C[:,1:]).
void symm__partial_range(int m, int n, double alpha, double beta,
                         double C[m][n], double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 1; j < n; j++) {
      double temp2 = 0;
      for (int k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
}
