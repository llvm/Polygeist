/* 2mm_jetson_wrapper.c — Jetson timing wrapper for kernel_2mm.
 *
 * kernel_2mm signature (polybench/linear-algebra/kernels/2mm):
 *   void kernel_2mm(int ni, int nj, int nk, int nl,
 *                   double alpha, double beta,
 *                   double tmp[NI][NJ], double A[NI][NK],
 *                   double B[NK][NJ], double C[NJ][NL], double D[NI][NL]);
 *
 * Bridges polybench's flat-pointer call to the MLIR-lowered impl which
 * takes 5 memref<?x?xf64> args expanded to (ptr, ptr, offset, size×2,
 * stride×2) — 7 args per matrix.
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_2mm_impl(
    int ni, int nj, int nk, int nl,
    double alpha, double beta,
    double *tmp_b, double *tmp_a, int64_t tmp_o, int64_t tmp_s0, int64_t tmp_s1, int64_t tmp_st0, int64_t tmp_st1,
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    double *B_b, double *B_a, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1,
    double *C_b, double *C_a, int64_t C_o, int64_t C_s0, int64_t C_s1, int64_t C_st0, int64_t C_st1,
    double *D_b, double *D_a, int64_t D_o, int64_t D_s0, int64_t D_s1, int64_t D_st0, int64_t D_st1);

extern void  polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_2mm(int ni, int nj, int nk, int nl,
                double alpha, double beta,
                double *tmp, double *A, double *B,
                double *C, double *D) {
  polygeist_cublas_time_begin();
  kernel_2mm_impl(ni, nj, nk, nl, alpha, beta,
                  tmp, tmp, 0, ni, nj, nj, 1,
                  A,   A,   0, ni, nk, nk, 1,
                  B,   B,   0, nk, nj, nj, 1,
                  C,   C,   0, nj, nl, nl, 1,
                  D,   D,   0, ni, nl, nl, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_2mm ni=%d nj=%d nk=%d nl=%d  %.3f ms\n",
          ni, nj, nk, nl, ms);
}
