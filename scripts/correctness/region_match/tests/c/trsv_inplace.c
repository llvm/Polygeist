/* b aliasing x: the right-hand side and the solution share one array
 * (in-place solve).  Expected: matched in place (X_in = X_out = x), flagged,
 * no copy emitted. */
void trsv_inplace(int n, double L[n][n], double x[n]) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
