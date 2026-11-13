//@ transform: compound_assign
void bicg__compound_assign(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = 0; i < m; i++)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    q[i] = 0.0;
    for (int j = 0; j < m; j++) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  }
}
