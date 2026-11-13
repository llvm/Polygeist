//@ transform: temp_scalar
void mvt__temp_scalar(int n, double x1[n], double x2[n], double y_1[n], double y_2[n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    double t = x1[i];
    for (int j = 0; j < n; j++)
      t = t + A[i][j] * y_1[j];
    x1[i] = t;
  }
  for (int i = 0; i < n; i++) {
    double t = x2[i];
    for (int j = 0; j < n; j++)
      t = t + A[j][i] * y_2[j];
    x2[i] = t;
  }
}
