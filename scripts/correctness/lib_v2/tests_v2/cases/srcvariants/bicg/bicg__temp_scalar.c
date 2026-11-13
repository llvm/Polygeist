//@ transform: temp_scalar
void bicg__temp_scalar(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = 0; i < m; i++)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    double qi = 0.0;
    double ri = r[i];
    for (int j = 0; j < m; j++) {
      s[j] = s[j] + ri * A[i][j];
      qi = qi + A[i][j] * p[j];
    }
    q[i] = qi;
  }
}
