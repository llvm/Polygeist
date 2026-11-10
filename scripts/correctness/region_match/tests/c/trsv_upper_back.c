/* Positive: row-major upper-triangular backward substitution (U x = b),
 * reversed loops. */
void trsv_upper_back(int n, double U[n][n], double x[n], double b[n]) {
  for (int i = n - 1; i >= 0; i--) {
    double t = b[i];
    for (int j = i + 1; j < n; j++)
      t -= U[i][j] * x[j];
    x[i] = t / U[i][i];
  }
}
