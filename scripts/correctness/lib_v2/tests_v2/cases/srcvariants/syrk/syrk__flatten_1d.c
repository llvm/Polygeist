//@ transform: flatten_1d
//@ extents: C=n*n, A=n*m
void syrk__flatten_1d(int n, int m, double alpha, double beta, double *C, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i * n + j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i * n + j] += alpha * A[i * m + k] * A[j * m + k];
  }
}
