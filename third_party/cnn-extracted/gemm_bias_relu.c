/* gemm_bias_relu.c — fused matmul + bias + relu, transformer FFN shape.
 *
 *   C[m,n] = relu(sum_k A[m,k] * B[k,n] + bias[n])
 *
 * Routes to cublasLt's CUBLASLT_EPILOGUE_RELU_BIAS for a single fused call.
 */
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
# define M  64
# define N  64
# define K  64
#elif defined(LARGE_DATASET)
# define M  2048
# define N  2048
# define K  2048
#else
# define M  64
# define N  64
# define K  64
#endif

void kernel_gemm_bias_relu(
    DATA_TYPE A[M][K],
    DATA_TYPE B[K][N],
    DATA_TYPE bias[N],
    DATA_TYPE C[M][N]) {
  int m, n, k;

  #pragma scop
  /* (1) Init: C = 0 */
  for (m = 0; m < M; ++m)
    for (n = 0; n < N; ++n)
      C[m][n] = 0;

  /* (2) Matmul: C += A * B */
  for (m = 0; m < M; ++m)
    for (n = 0; n < N; ++n)
      for (k = 0; k < K; ++k)
        C[m][n] += A[m][k] * B[k][n];

  /* (3) Bias add (per column, broadcast over rows) */
  for (m = 0; m < M; ++m)
    for (n = 0; n < N; ++n)
      C[m][n] += bias[n];

  /* (4) ReLU (ternary form) */
  for (m = 0; m < M; ++m)
    for (n = 0; n < N; ++n) {
      DATA_TYPE v = C[m][n];
      C[m][n] = (v > 0.0f) ? v : 0.0f;
    }
  #pragma endscop
}
