// Reduction stored into one row of an output array (GEMM-inner-loop pattern).
void rowsum(int m, int n, double *A, double *out) {
  for (int i = 0; i < m; i++) {
    double s = 0;
    for (int j = 0; j < n; j++) s += A[i*n + j];
    out[i] = s;
  }
}
