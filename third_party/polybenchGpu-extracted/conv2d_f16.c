// conv2d_f16.c — half-precision (_Float16) variant of the extracted conv2d
// kernel. Same 3x3 polybench filter as conv2d.c but in _Float16 instead of
// double. Used to validate Phase 2 FP16 generalization: the matcher
// fingerprints any half-dtype conv body, the rewriter emits a `_f16`-suffixed
// launch symbol, ABI lowering dispatches to the f16 runtime shim.
//
// Weights use the same 0.X polybench filter as conv2d.c. _Float16 has only
// ~3 decimal digits of precision, so a literal like 0.2f16 isn't exactly
// 0.2 — the bit-exact validator must be tolerant of that. Use the CPU stub
// (which accumulates in float and downcasts on store) as the reference; the
// CUDA path also uses FP32 internal accumulation so both should agree.

#ifndef NI
#define NI 256
#endif
#ifndef NJ
#define NJ 256
#endif

void kernel_conv2d(int ni, int nj,
                   _Float16 A[NI][NJ], _Float16 B[NI][NJ]) {
  int i, j;
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j) {
      B[i][j] = (_Float16)0.2 * A[i-1][j-1] + (_Float16)0.5 * A[i-1][j]
              + (_Float16)-0.8 * A[i-1][j+1]
              + (_Float16)-0.3 * A[ i ][j-1] + (_Float16)0.6 * A[ i ][j]
              + (_Float16)-0.9 * A[ i ][j+1]
              + (_Float16)0.4 * A[i+1][j-1] + (_Float16)0.7 * A[i+1][j]
              + (_Float16)0.1 * A[i+1][j+1];
    }
}
