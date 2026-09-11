/*
 * Computational region extracted from Parboil sgemm/src/base/sgemm_kernel.cc.
 *
 * Only the C++ diagnostic stream operations are omitted.  The argument ABI,
 * guards, loop nest, array indexing, arithmetic, and update of C are retained.
 */
void parboil_basic_sgemm(
    char transa, char transb, int m, int n, int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta,
    float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n'))
    return;
  if ((transb != 'T') && (transb != 't'))
    return;

  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
    }
  }
}
