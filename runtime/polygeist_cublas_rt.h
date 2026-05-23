// polygeist_cublas_rt.h — runtime shim ABI for the
// `--lower-kernel-launch-to-cublas` pass.
//
// The pass emits `func.call` ops targeting these C functions. The functions
// are implemented in two flavours:
//   * polygeist_cublas_rt_cpu.c   — reference CPU implementation (no CUDA).
//                                    Used for correctness validation on
//                                    machines without a GPU.
//   * polygeist_cublas_rt_cuda.c  — real cuBLAS implementation. Used on
//                                    Jetson / x86 + NVIDIA GPU.
// Link exactly one of them into the executable.
//
// All matrices are ROW-MAJOR f64. Leading dimensions are in elements
// (not bytes). The CUDA backend internally does the row↔col-major dance
// (compute Cᵀ = BᵀAᵀ via operand swap) so callers can stay row-major.
//
// Sizes are passed as int32_t because that matches cuBLAS's signature.

#ifndef POLYGEIST_CUBLAS_RT_H
#define POLYGEIST_CUBLAS_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle. Call init() once before any kernel calls; destroy() at exit.
// On CPU these are no-ops; on CUDA they create a cublasHandle_t + stream.
void polygeist_cublas_init(void);
void polygeist_cublas_destroy(void);

// GEMM (cublasDgemm equivalent, row-major):
//   C = alpha * A * B + beta * C
// where A is MxK, B is KxN, C is MxN.
//
// For non-transposed inputs at row-major:
//   lda = K, ldb = N, ldc = N.
//
// On CUDA: copies A/B/C H→D, calls cublasDgemm with operand swap to handle
// the row→col-major transpose, copies C D→H, frees device buffers. Each call
// is fully synchronous; device-residency hoisting is a follow-up.
void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc);

// Per-call CUDA-event timing (CUDA backend only — CPU stub returns 0.0).
// Pair with polygeist_cublas_time_begin / polygeist_cublas_time_end around
// a sequence of kernel calls.
void  polygeist_cublas_time_begin(void);
double polygeist_cublas_time_end_ms(void);  // returns ms since last begin

#ifdef __cplusplus
}
#endif

#endif  // POLYGEIST_CUBLAS_RT_H
