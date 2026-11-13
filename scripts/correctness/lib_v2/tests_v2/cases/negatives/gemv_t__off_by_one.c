//@ routine: gemv_t
//@ kind: off_by_one
//@ expect: nomatch
// beta scaling skips the last element (j < n-1) while all n are accumulated.
void gemv_t__off_by_one(int m, int n, double alpha, double beta,
                        double A[m][n], double x[m], double y[n]) {
  for (int j = 0; j < n - 1; j++)
    y[j] = beta * y[j];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      y[j] += alpha * A[i][j] * x[i];
}
