//@ transform: ptr_arith
void bicg__ptr_arith(int m, int n, double A[n][m], double s[m], double q[n], double p[m], double r[n]) {
  for (int i = 0; i < m; i++)
    *(s + i) = 0;
  for (int i = 0; i < n; i++) {
    double *a = A[i];
    *(q + i) = 0.0;
    for (int j = 0; j < m; j++) {
      *(s + j) = *(s + j) + *(r + i) * *(a + j);
      *(q + i) = *(q + i) + *(a + j) * *(p + j);
    }
  }
}
