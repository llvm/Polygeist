// conv2d_f32.c — single-precision (float) variant of the extracted conv2d
// kernel. Same 3x3 polybench filter as conv2d.c but in float instead of
// double. Used to validate Phase 2 of the cuDNN conv generalization —
// matcher fingerprints any float-dtype conv body, emits a dtype-suffixed
// launch symbol, ABI lowering dispatches to the f32 runtime shim.

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

void kernel_conv2d(int ni, int nj,
                   float A[NI][NJ], float B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] =  0.2f * A[i-1][j-1] +  0.5f * A[i-1][j] + -0.8f * A[i-1][j+1]
              + -0.3f * A[ i ][j-1] +  0.6f * A[ i ][j] + -0.9f * A[ i ][j+1]
              +  0.4f * A[i+1][j-1] +  0.7f * A[i+1][j] +  0.1f * A[i+1][j+1];
    }
}
