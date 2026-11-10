//@ transform: split_reduction
void bicg__split_reduction(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = 0; i < m; i++)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    double q0 = 0.0, q1 = 0.0;
    for (int j = 0; j < m / 2; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q0 = q0 + A[i][j] * p[j];
    }
    for (int j = m / 2; j < m; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q1 = q1 + A[i][j] * p[j];
    }
    q[i] = q0 + q1;
  }
}
