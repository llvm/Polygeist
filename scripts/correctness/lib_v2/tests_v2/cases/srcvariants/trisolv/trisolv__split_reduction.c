//@ transform: split_reduction
void trisolv__split_reduction(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    double s0 = 0.0, s1 = 0.0;
    for (int j = 0; j < i / 2; j++)
      s0 += L[i][j] * x[j];
    for (int j = i / 2; j < i; j++)
      s1 += L[i][j] * x[j];
    x[i] = (b[i] - (s0 + s1)) / L[i][i];
  }
}
