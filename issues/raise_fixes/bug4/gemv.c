/* Bug 4 minimal reproducer: y = A*x.  Under the run_kernel_e2e.sh --debuf
 * recipe (raise + --lower-polygeist-submap, then plain --linalg-debufferize)
 * the 1-D identity output view is lowered to memref.reinterpret_cast, which
 * debufferize cannot convert; it used to spin forever instead of bailing. */
void gemv(int m, int n, double A[m][n], double x[n], double y[m]) {
  for (int i = 0; i < m; i++) {
    y[i] = 0.0;
    for (int j = 0; j < n; j++)
      y[i] += A[i][j] * x[j];
  }
}
