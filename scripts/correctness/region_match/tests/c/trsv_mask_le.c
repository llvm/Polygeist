/* Negative: j <= i includes the diagonal in the update, so x[i] reads its own
 * partially-updated value (not a triangular solve). */
void trsv_mask_le(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j <= i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
