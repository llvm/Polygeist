/* Positive: unit-diagonal forward substitution (no divide).  Must match the
 * DIAG='U' spec, never DIAG='N'. */
void trsv_unit(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
  }
}
