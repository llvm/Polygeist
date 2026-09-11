/* syrk_jetson_wrapper.c — Jetson timing wrapper.
 *
 * Bridges polybenchGpu's kernel_syrk(int ni, int nj, double alpha, double beta,
 * double C[NI][NI], double A[NI][NJ]) signature to the MLIR-lowered
 * kernel_syrk_impl that takes bare memref descriptor args.
 *
 * Wraps the call with polygeist_cublas_time_begin/end_ms so we get a per-call
 * timing print on stderr. On the CUDA runtime, timing uses cudaEvents.
 *
 * Matches gemm_jetson_wrapper.c structure.
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_syrk_impl(
    int ni, int nj, double alpha, double beta,
    double *C_base, double *C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1,
    double *A_base, double *A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_syrk(int ni, int nj, double alpha, double beta,
                 double *C, double *A) {
  polygeist_cublas_time_begin();
  kernel_syrk_impl(ni, nj, alpha, beta,
                   C, C, 0, ni, ni, ni, 1,
                   A, A, 0, ni, nj, nj, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_syrk ni=%d nj=%d  %.3f ms\n",
          ni, nj, ms);
}
