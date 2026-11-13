//@ transform: reverse
// The dot-product loop runs backwards.
void trisolv__reverse(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = i - 1; j >= 0; j--)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
