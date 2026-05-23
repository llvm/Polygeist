/* conv2d_jetson_wrapper_dtype.c — dtype-parameterized timing wrapper.
 *
 * Compile with -DCTYPE=<scalar C type>. After MLIR lowering the kernel is
 * `kernel_conv2d_impl` with the memref descriptor expansion (7 args per
 * 2D memref).
 */
#include <stdint.h>
#include <stdio.h>

#ifndef CTYPE
#define CTYPE double
#endif

extern void kernel_conv2d_impl(
    int ni, int nj,
    CTYPE *A_b, CTYPE *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    CTYPE *B_b, CTYPE *B_a, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1);

extern void  polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_conv2d(int ni, int nj, CTYPE *A, CTYPE *B) {
  polygeist_cublas_time_begin();
  kernel_conv2d_impl(ni, nj,
                     A, A, 0, ni, nj, nj, 1,
                     B, B, 0, ni, nj, nj, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_conv2d ni=%d nj=%d  %.3f ms\n",
          ni, nj, ms);
}
