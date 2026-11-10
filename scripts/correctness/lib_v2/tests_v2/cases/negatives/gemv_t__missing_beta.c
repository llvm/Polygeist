//@ routine: gemv_t
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
void gemv_t__missing_beta(int m, int n, double alpha, double beta,
                          double A[m][n], double x[m], double y[n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * x[i];
}
