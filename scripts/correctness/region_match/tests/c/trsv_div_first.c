/* Negative: divides before subtracting, x[i] = b[i]/L[i][i] - sum(...). */
void trsv_div_first(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i] / L[i][i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
  }
}
