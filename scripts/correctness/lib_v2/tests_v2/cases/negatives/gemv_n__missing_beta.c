//@ routine: gemv_n
//@ kind: missing_beta
//@ expect: nomatch_or_equivalent
void gemv_n__missing_beta(int m, int n, double alpha, double beta,
                          double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[i] += alpha * A[i][j] * x[j];
}
