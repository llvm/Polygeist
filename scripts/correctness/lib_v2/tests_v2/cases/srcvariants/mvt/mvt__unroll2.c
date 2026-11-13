//@ transform: unroll2
void mvt__unroll2(int n, double x1[n], double x2[n], double y_1[n], double y_2[n], double A[n][n]) {
  for (int i = 0; i < n; i++) {
    int j = 0;
    for (; j + 1 < n; j += 2) {
      x1[i] = x1[i] + A[i][j] * y_1[j];
      x1[i] = x1[i] + A[i][j + 1] * y_1[j + 1];
    }
    for (; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
}
