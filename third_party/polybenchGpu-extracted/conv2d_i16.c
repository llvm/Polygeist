// conv2d_i16.c — int16_t variant of the extracted conv2d kernel. Tests the
// INT16 path: matcher binds the int conv body, the rewriter emits
// @cudnnConvolution2D_9tap_i16, and the ABI lowering routes to the i16
// shim. The shim itself upcasts to int32 internally because cuDNN has no
// native i16 convolution.

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

void kernel_conv2d(int ni, int nj,
                   short A[NI][NJ], short B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] = (short)( 2 * A[i-1][j-1] +  5 * A[i-1][j] + -8 * A[i-1][j+1]
                       + -3 * A[ i ][j-1] +  6 * A[ i ][j] + -9 * A[ i ][j+1]
                       +  4 * A[i+1][j-1] +  7 * A[i+1][j] +  3 * A[i+1][j+1]);
    }
}
