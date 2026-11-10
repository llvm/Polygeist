/* Negative: LDL^T-like factorisation (unit lower L, diagonal D, no sqrt). */
void ldlt(int n, double A[n][n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k] * A[k][k];
      A[i][j] /= A[j][j];
    }
    for (int k = 0; k < i; k++)
      A[i][i] -= A[i][k] * A[i][k] * A[k][k];
  }
}
