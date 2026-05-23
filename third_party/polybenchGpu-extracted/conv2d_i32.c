// conv2d_i32.c — int32_t variant of the extracted conv2d kernel. Same 3x3
// stencil shape as conv2d.c but with integer weights and inputs. Used to
// validate the Phase-2 INT32 path: matcher recognises arith.muli/addi,
// emits @cudnnConvolution2D_9tap_i32, ABI lowering dispatches to
// polygeist_cudnn_conv2d_3x3_i32 (cuDNN's CUDNN_DATA_INT32 path).
//
// Weights chosen so 9-tap sums don't overflow int32 for reasonable input
// magnitudes — small ints with mixed signs.

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

void kernel_conv2d(int ni, int nj,
                   int A[NI][NJ], int B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] =  2 * A[i-1][j-1] +  5 * A[i-1][j] + -8 * A[i-1][j+1]
              + -3 * A[ i ][j-1] +  6 * A[ i ][j] + -9 * A[ i ][j+1]
              +  4 * A[i+1][j-1] +  7 * A[i+1][j] +  3 * A[i+1][j+1];
    }
}
