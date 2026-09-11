/*
 * Algebraically normalized form of the extracted Parboil SGEMM region.
 * This is a diagnostic fixture: unlike parboil_sgemm.c, it changes the loop
 * schedule by separating beta scaling from alpha*A*B accumulation.
 */
void parboil_basic_sgemm_normalized(
    int m, int n, int k, float alpha, const float *A, int lda,
    const float *B, int ldb, float beta, float *C, int ldc) {
  for (int mm = 0; mm < m; ++mm)
    for (int nn = 0; nn < n; ++nn)
      C[mm + nn * ldc] = beta * C[mm + nn * ldc];

  for (int mm = 0; mm < m; ++mm)
    for (int nn = 0; nn < n; ++nn)
      for (int i = 0; i < k; ++i)
        C[mm + nn * ldc] += alpha * A[mm + i * lda] * B[nn + i * ldb];
}
