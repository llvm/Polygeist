/* conv2d_i8.c — int8_t variant of the extracted polybenchGpu conv2d kernel.
 * Tests the INT8 path: matcher binds the int conv body via its dtype-
 * agnostic encoding, the rewriter sniffs the operand element type
 * (i8) and emits @cudnnConvolution2D_9tap_i8, and the ABI lowering
 * routes to the polygeist_pva_conv2d_3x3_i8 runtime shim (NOT to
 * cuDNN — cuDNN doesn't accept INT8 standalone conv, but PVA Solutions'
 * cupva-backed pvaConv2d does).
 *
 * Weights are the polybench 9-tap pattern scaled to INT8 range. Product
 * widths (8b weight * 8b pixel) need a wider accumulator — the C body
 * here lets cgeist emit `arith.muli i8` plus implicit `arith.extsi` to a
 * wider compute type, which the matcher's transparent-cast handling
 * absorbs.
 */

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

/* signed char ≡ int8_t in the polybench style — keeps cgeist happy
 * without needing <stdint.h>. */
void kernel_conv2d(int ni, int nj,
                   signed char A[NI][NJ], signed char B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] = (signed char)(
            2 * A[i-1][j-1] +  5 * A[i-1][j] + -8 * A[i-1][j+1]
         + -3 * A[ i ][j-1] +  6 * A[ i ][j] + -9 * A[ i ][j+1]
         +  4 * A[i+1][j-1] +  7 * A[i+1][j] +  3 * A[i+1][j+1]);
    }
}
