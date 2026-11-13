/* gemm_jetson_wrapper.c — Jetson timing wrapper.
 *
 * Same shape as gemm_wrapper.c (bridges PolyBench's kernel_gemm signature
 * to the MLIR-lowered kernel_gemm_impl with bare memref descriptor args),
 * but additionally wraps the call with polygeist_cublas_time_begin/end_ms
 * so we get a per-call timing print on the Jetson.
 *
 * On the CUDA runtime, timing uses cudaEvents (GPU time). On the CPU stub,
 * it uses CLOCK_MONOTONIC wall-clock. Either way it goes to stderr so
 * stdout numerics stay clean for diff against the reference.
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_gemm_impl(
    int ni, int nj, int nk, double alpha, double beta,
    double *C_base, double *C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    double *A_base, double *A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    double *B_base, double *B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1);

extern void  polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                 double *C, double *A, double *B) {
  polygeist_cublas_time_begin();
  kernel_gemm_impl(ni, nj, nk, alpha, beta,
                   C, C, 0, ni, nj, nj, 1,
                   A, A, 0, ni, nk, nk, 1,
                   B, B, 0, nk, nj, nj, 1);
  double ms = polygeist_cublas_time_end_ms();
  /* stderr because PolyBench dumps the result array to stderr too; we
   * prefix with a sentinel so test diff scripts can grep it out. */
  fprintf(stderr, "POLYGEIST_TIMING: kernel_gemm ni=%d nj=%d nk=%d  %.3f ms\n",
          ni, nj, nk, ms);
}
