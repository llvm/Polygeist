//@ routine: gemv_t
//@ kind: extra_term
//@ expect: nomatch
void gemv_t__extra_term(int m, int n, double alpha, double beta,
                        double A[m][n], double x[m], double y[n]) {
  for (int j = 0; j < n; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * x[i] + A[i][j];
}
