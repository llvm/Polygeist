//@ routine: gemv_n
//@ kind: extra_term
//@ expect: nomatch
void gemv_n__extra_term(int m, int n, double alpha, double beta,
                        double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++) {
    y[i] = beta * y[i];
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * x[j] + x[j];
  }
}
