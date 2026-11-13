//@ transform: interchange
// The two products are separated; s = A^T r runs j-outer.
void bicg__interchange(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int j = 0; j < m; j++) {
    s[j] = 0;
    for (int i = 0; i < n; i++)
      s[j] = s[j] + r[i] * A[i][j];
  }
  for (int i = 0; i < n; i++) {
    q[i] = 0.0;
    for (int j = 0; j < m; j++)
      q[i] = q[i] + A[i][j] * p[j];
  }
}
