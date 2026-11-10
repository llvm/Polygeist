/* Positive: column-oriented (axpy-form) forward substitution, row-major L,
 * separate right-hand side.  Same math as PolyBench trisolv, different loop
 * order and IR shape. */
void trsv_axpy(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++)
    x[i] = b[i];
  for (int j = 0; j < n; j++) {
    x[j] = x[j] / L[j][j];
    for (int i = j + 1; i < n; i++)
      x[i] = x[i] - L[i][j] * x[j];
  }
}
