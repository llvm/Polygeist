/* Jetson harness for AᵀA via syrk-alias discriminator. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define M 2048
# define K 2048
#elif defined(MINI_DATASET)
# define M 64
# define K 64
#endif
#ifndef M
# define M 64
#endif
#ifndef K
# define K 64
#endif

extern void kernel_ata_gemm_impl(
    float *A_b, float *A_a, int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_t0, int64_t A_t1,
    float *C_b, float *C_a, int64_t C_o,
    int64_t C_s0, int64_t C_s1, int64_t C_t0, int64_t C_t1);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *C) {
  polygeist_cublas_time_begin();
  kernel_ata_gemm_impl(
      A, A, 0, (int64_t)K, (int64_t)M, (int64_t)M, 1,
      C, C, 0, (int64_t)M, (int64_t)M, (int64_t)M, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: ata_gemm M=%d K=%d  %.3f ms\n",
      M, K, ms);
}

int main(void) {
  size_t nA = (size_t)K * M;
  size_t nC = (size_t)M * M;
  float *A = (float *)malloc(nA * sizeof(float));
  float *C = (float *)malloc(nC * sizeof(float));
  if (!A || !C) { fprintf(stderr, "alloc failed\n"); return 1; }

  for (size_t k = 0; k < nA; ++k)
    A[k] = (float)((k * 17) % 31) / 31.0f - 0.5f;
  memset(C, 0, nC * sizeof(float));

  run_kernel(A, C);

  double sum = 0;
  for (size_t k = 0; k < nC; ++k) sum += C[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, nC);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nC; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", C[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(C);
  return 0;
}
