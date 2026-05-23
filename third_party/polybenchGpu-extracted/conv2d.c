// conv2d.c — extracted from polybenchGpu/OpenMP/stencils/convolution-2d/.
//
// Why this extraction exists: the original polybenchGpu file mixes
// kernel_conv2d + init_array + main + print_array in one TU. cgeist
// inlines everything into main; the optimizer then notices init_array
// writes A[i][j] = (i+j)/nj (a constant function of indices) and
// constant-folds the entire conv2d body — the lifted linalg.generic
// ends up with NO ins(A), just synthesises B[i,j] = closed-form
// function of indices. That bypass makes the matcher unable to
// fingerprint a conv2d shape (no input operand to match against).
//
// This extraction breaks the inlining chain: the function is alone in
// its TU, takes A and B as explicit parameters, and uses fixed sizes
// baked in via #define so the loop bounds are constant. The lift
// produces a clean linalg.generic with ins(A) outs(B) and the matcher
// can recognise it.
//
// Mirrors third_party/NPB-polybenchified/ in spirit and convention.

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

// 9-tap 3x3 stencil, weights from polybenchGpu's original kernel_conv2d.
void kernel_conv2d(int ni, int nj,
                   double A[NI][NJ], double B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] =  0.2 * A[i-1][j-1] +  0.5 * A[i-1][j] + -0.8 * A[i-1][j+1]
              + -0.3 * A[ i ][j-1] +  0.6 * A[ i ][j] + -0.9 * A[ i ][j+1]
              +  0.4 * A[i+1][j-1] +  0.7 * A[i+1][j] +  0.1 * A[i+1][j+1];
    }
}
