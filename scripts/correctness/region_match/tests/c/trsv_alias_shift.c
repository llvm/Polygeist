/* Negative (b partially aliasing x): the right-hand side is the same buffer
 * shifted by one element, b = xb + 1.  The read of xb[i+1] is an input value
 * at a shifted index; no spec binding exists -> must not match. */
void trsv_alias_shift(int n, double L[n][n], double xb[n + 1]) {
  for (int i = 0; i < n; i++) {
    xb[i] = xb[i + 1];
    for (int j = 0; j < i; j++)
      xb[i] -= L[i][j] * xb[j];
    xb[i] = xb[i] / L[i][i];
  }
}
