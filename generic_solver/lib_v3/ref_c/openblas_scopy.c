/* OpenBLAS 0.3.31, BSD-3-Clause; see ../upstream/openblas-0.3.31/LICENSE.
 * Genuine portable C kernel/arm/copy.c. FLOAT=float, BLASLONG=long;
 * specialize unit increments / alpha=1 and positive extents as declared.
 * Pointer column increments become equivalent VLA indexing, same loop order.
 * Status return and invalid-input early exits are omitted under those assumptions.
 */
void NETLIB_SPEC(int n, float x[], float y[]) {
  long i=0, ix=0, iy=0;
  while(i<n) { y[iy]=x[ix]; ix+=1; iy+=1; i++; }
}
