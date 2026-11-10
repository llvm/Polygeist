//@ transform: tiling
void bicg__tiling(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = 0; i < m; i++)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    q[i] = 0.0;
    for (int jj = 0; jj < m; jj += 4) {
      int je = jj + 4 < m ? jj + 4 : m;
      for (int j = jj; j < je; j++) {
        s[j] = s[j] + r[i] * A[i][j];
        q[i] = q[i] + A[i][j] * p[j];
      }
    }
  }
}
