/* Jetson harness for fused gemm + bias + relu (cublasLt epilogue). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define M 2048
# define N 2048
# define K 2048
#elif defined(MINI_DATASET)
# define M 64
# define N 64
# define K 64
#endif
#ifndef M
# define M 64
#endif
#ifndef N
# define N 64
#endif
#ifndef K
# define K 64
#endif

extern void kernel_gemm_bias_relu_impl(
    float *A_b, float *A_a, int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_t0, int64_t A_t1,
    float *B_b, float *B_a, int64_t B_o,
    int64_t B_s0, int64_t B_s1, int64_t B_t0, int64_t B_t1,
    float *Bi_b, float *Bi_a, int64_t Bi_o, int64_t Bi_sz, int64_t Bi_st,
    float *C_b, float *C_a, int64_t C_o,
    int64_t C_s0, int64_t C_s1, int64_t C_t0, int64_t C_t1);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *B, float *bias, float *C) {
  polygeist_cublas_time_begin();
  kernel_gemm_bias_relu_impl(
      A, A, 0, (int64_t)M, (int64_t)K, (int64_t)K, 1,
      B, B, 0, (int64_t)K, (int64_t)N, (int64_t)N, 1,
      bias, bias, 0, (int64_t)N, 1,
      C, C, 0, (int64_t)M, (int64_t)N, (int64_t)N, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: gemm_bias_relu M=%d N=%d K=%d  %.3f ms\n",
      M, N, K, ms);
}

int main(void) {
  size_t nA = (size_t)M*K, nB = (size_t)K*N, nC = (size_t)M*N;
  float *A    = (float *)malloc(nA * sizeof(float));
  float *B    = (float *)malloc(nB * sizeof(float));
  float *C    = (float *)malloc(nC * sizeof(float));
  float *bias = (float *)malloc(N * sizeof(float));
  if (!A || !B || !C || !bias) { fprintf(stderr, "alloc failed\n"); return 1; }

  for (size_t k = 0; k < nA; ++k)
    A[k] = (float)((k * 17) % 31) / 31.0f - 0.5f;
  for (size_t k = 0; k < nB; ++k)
    B[k] = (float)((k * 23) % 37) / 37.0f - 0.5f;
  for (int n = 0; n < N; ++n)
    bias[n] = 0.01f * (float)n - 0.1f;
  memset(C, 0, nC * sizeof(float));

  run_kernel(A, B, bias, C);

  double sum = 0; size_t nz = 0;
  for (size_t k = 0; k < nC; ++k) { sum += C[k]; if (C[k] == 0.0f) ++nz; }
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems, %zu zeroed (%.1f%%)\n",
          sum, nC, nz, 100.0 * (double)nz / (double)nC);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nC; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", C[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(B); free(C); free(bias);
  return 0;
}
