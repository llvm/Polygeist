/* Negative: divide operands swapped (L[i][i] / x[i]). */
void trsv_swapdiv(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = L[i][i] / x[i];
  }
}
