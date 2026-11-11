/* OpenBLAS 0.3.31, BSD-3-Clause; see ../upstream/openblas-0.3.31/LICENSE.
 * Genuine portable C kernel/arm/omatcopy_cn.c. FLOAT=float, BLASLONG=long;
 * specialize unit increments / alpha=1 and positive extents as declared.
 * Pointer column increments become equivalent VLA indexing, same loop order.
 * Status return and invalid-input early exits are omitted under those assumptions.
 */
void NETLIB_SPEC(int m, int n, int lda, float A[][lda], int ldc, float C[][ldc]) {
  for(long i=0;i<n;i++)
    for(long j=0;j<m;j++) C[i][j]=A[i][j];
}
