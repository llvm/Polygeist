/* OpenBLAS 0.3.31, BSD-3-Clause; see ../upstream/openblas-0.3.31/LICENSE.
 * Genuine portable C kernel/arm/omatcopy_{cn,ct}.c alpha==1 branches.
 * FLOAT=float, BLASLONG=long; positive extents and nonoverlap are contracts.
 * Pointer column increments become identical VLA indices, same loop order.
 * m,n name the target API output rows/columns: upstream rows=m,cols=n for
 * no transpose; upstream rows=n,cols=m for transpose. No arithmetic changes.
 * Status return and invalid-input early exits omitted under the contract.
 */
#ifndef TRANS
#define TRANS 'N'
#endif
void NETLIB_SPEC(int m, int n, int lda, float A[][lda], int ldc, float C[][ldc]) {
#if TRANS == 'N'
  for(long i=0;i<n;i++)
    for(long j=0;j<m;j++) C[i][j]=A[i][j];
#elif TRANS == 'T'
  for(long i=0;i<m;i++)
    for(long j=0;j<n;j++) C[j][i]=A[i][j];
#else
#error "unsupported transpose mode"
#endif
}
