/* ata_gemm.c — AᵀA, a Gram-matrix shape that LOOKS like a gemm to the
 * matcher's body unifier but happens to read the same tensor twice.
 *
 *   C[m, n] = sum_k A[k, m] * A[k, n]      // AᵀA — symmetric output
 *
 * The matcher's discriminator (post-unify check on operand aliasing)
 * should detect that both ins of the matched gemm body resolve to the
 * same underlying tensor and route to cublasDsyrk (half the flops:
 * writes only the upper triangle, beta=0).
 */
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
# define M 64
# define K 64
#elif defined(LARGE_DATASET)
# define M 2048
# define K 2048
#else
# define M 64
# define K 64
#endif

/* C = AᵀA. A is K×M, C is M×M, symmetric. Explicit init + accumulate
 * form: that's what's idiomatic in real-world gemm-shaped C code, and
 * is what the matcher's 2-step gemm composition expects. The
 * cublasSsyrk shim overwrites C with β=0, so the preceding memset is
 * mathematically redundant — the lowering pass detects the
 * "memset_zero_2D launch immediately preceding a syrk_alias launch on
 * the same output base" pattern and erases the memset. */
void kernel_ata_gemm(DATA_TYPE A[K][M], DATA_TYPE C[M][M]) {
  int m, n, k;

  #pragma scop
  for (m = 0; m < M; ++m)
    for (n = 0; n < M; ++n)
      C[m][n] = 0;

  for (m = 0; m < M; ++m)
    for (n = 0; n < M; ++n)
      for (k = 0; k < K; ++k)
        C[m][n] += A[k][m] * A[k][n];
  #pragma endscop
}
