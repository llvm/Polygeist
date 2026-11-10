//@ transform: split_reduction
void mvt__split_reduction(int n, double x1[n], double x2[n], double y_1[n], double y_2[n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    double s0 = 0.0, s1 = 0.0;
    for (int j = 0; j < n / 2; j++)
      s0 = s0 + A[i][j] * y_1[j];
    for (int j = n / 2; j < n; j++)
      s1 = s1 + A[i][j] * y_1[j];
    x1[i] = x1[i] + (s0 + s1);
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
}
