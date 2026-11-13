//@ transform: compound_assign
void mvt__compound_assign(int n, double x1[n], double x2[n], double y_1[n], double y_2[n], double A[n][n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x1[i] += A[i][j] * y_1[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x2[i] += A[j][i] * y_2[j];
}
