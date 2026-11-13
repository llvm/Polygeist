//@ transform: reverse
void bicg__reverse(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = m - 1; i >= 0; i--)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    q[i] = 0.0;
    for (int j = m - 1; j >= 0; j--) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
}
