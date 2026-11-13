//@ transform: unroll2
void trisolv__unroll2(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    int j = 0;
    for (; j + 1 < i; j += 2) {
      x[i] -= L[i][j] * x[j];
      x[i] -= L[i][j + 1] * x[j + 1];
    }
    for (; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
