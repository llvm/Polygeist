//@ routine: trmm
//@ kind: positive_control
//@ expect: match
// PolyBench trmm: B := alpha*A^T*B with A unit lower triangular (row-major).
void trmm__pos(int m, int n, double alpha, double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}
