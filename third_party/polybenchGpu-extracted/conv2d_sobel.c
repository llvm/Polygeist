// conv2d_sobel.c — Sobel-X-like edge filter, scaled by 1.5 so the matcher
// validation isn't confused by clang's `1.0 * x → x` identity-fold (which
// removes mulf ops for unit weights — a separate generality gap tracked in
// project-cudnn-conv-pipeline-generality-gaps).
//
// Scaled Sobel-X filter:
//   [-1.5, 0,  1.5]    no 1.0 or -1.0 weights → mulf ops preserved
//   [-2.0, 0,  2.0]    0.0 weights are FINE (mulf-by-0 not identity-folded)
//   [-1.5, 0,  1.5]
//
// 5 distinct weights: -2.0, -1.5, 0.0, 1.5, 2.0. Used to prove the matcher
// surfaces arbitrary 3x3 weights (not just polybench's specific filter).

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

void kernel_conv2d(int ni, int nj,
                   double A[NI][NJ], double B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] = -1.5 * A[i-1][j-1] +  0.0 * A[i-1][j] +  1.5 * A[i-1][j+1]
             + -2.0 * A[ i ][j-1] +  0.0 * A[ i ][j] +  2.0 * A[ i ][j+1]
             + -1.5 * A[i+1][j-1] +  0.0 * A[i+1][j] +  1.5 * A[i+1][j+1];
    }
}
