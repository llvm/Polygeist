// polygeist_cublas_rt_cpu.c — reference CPU implementation of the runtime
// shim ABI. No CUDA dependency. Used for end-to-end correctness validation
// on machines without a GPU.
//
// The math is intentionally the slowest possible 3-loop gemm: the goal is
// to validate the lowering pass and the runtime call shape, not to be fast.

#include "polygeist_cublas_rt.h"

#include <stdio.h>
#include <time.h>

void polygeist_cublas_init(void) { /* no-op */ }
void polygeist_cublas_destroy(void) { /* no-op */ }

void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc) {
  // C[i,j] = alpha * sum_k A[i,k] * B[k,j] + beta * C[i,j]
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int32_t k = 0; k < K; ++k) {
        acc += A[(size_t)i * (size_t)lda + (size_t)k] *
               B[(size_t)k * (size_t)ldb + (size_t)j];
      }
      double *c = &C[(size_t)i * (size_t)ldc + (size_t)j];
      *c = alpha * acc + beta * (*c);
    }
  }
}

// CPU stub timing — wall-clock via clock_gettime(CLOCK_MONOTONIC). Useful
// for sanity but not for GPU perf numbers.

static struct timespec g_t0;

void polygeist_cublas_time_begin(void) {
  clock_gettime(CLOCK_MONOTONIC, &g_t0);
}

double polygeist_cublas_time_end_ms(void) {
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double dt_ns = (double)(t1.tv_sec - g_t0.tv_sec) * 1.0e9 +
                 (double)(t1.tv_nsec - g_t0.tv_nsec);
  return dt_ns / 1.0e6;
}
