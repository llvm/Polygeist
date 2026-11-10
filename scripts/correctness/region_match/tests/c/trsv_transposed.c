/* Negative for the PolyBench (lower, row-major) solve: reads L[j][i], i.e.
 * the upper triangle -- this is the transposed solve L^T-style system.  Must
 * not be recognised as netlib_dtrsv_utn. */
void trsv_transposed(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[j][i] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
