/* Positive: dot-form forward substitution through an explicit scalar temp. */
void trsv_temp(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    double t = b[i];
    for (int j = 0; j < i; j++)
      t -= L[i][j] * x[j];
    x[i] = t / L[i][i];
  }
}
