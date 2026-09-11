/* conv2d_jetson_wrapper.c — Jetson timing wrapper for extracted conv2d.
 *
 * The extracted kernel signature is:
 *   void kernel_conv2d(int ni, int nj, double A[NI][NJ], double B[NI][NJ]);
 *
 * After MLIR lowering it becomes kernel_conv2d_impl with the memref
 * descriptor expansion (each 2D memref unpacks into 7 args).
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_conv2d_impl(
    int ni, int nj,
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    double *B_b, double *B_a, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1);

extern void  polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_conv2d(int ni, int nj, double *A, double *B) {
  polygeist_cublas_time_begin();
  kernel_conv2d_impl(ni, nj,
                     A, A, 0, ni, nj, nj, 1,
                     B, B, 0, ni, nj, nj, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_conv2d ni=%d nj=%d  %.3f ms\n",
          ni, nj, ms);
}
