/* atax_jetson_wrapper.c — Jetson timing wrapper.
 *
 * polybenchGpu kernel_atax computes:
 *   tmp = A·x        (gemv)
 *   y   = Aᵀ·tmp     (gemv)
 *
 * Bridges polybenchGpu's kernel_atax(nx, ny, A, x, y, tmp) to the
 * MLIR-lowered kernel_atax_impl with memref-descriptor args. Per-call
 * timing on stderr.
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_atax_impl(
    int nx, int ny,
    /* A: 2D memref */
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    /* x: 1D memref */
    double *x_b, double *x_a, int64_t x_o, int64_t x_s, int64_t x_st,
    /* y: 1D memref */
    double *y_b, double *y_a, int64_t y_o, int64_t y_s, int64_t y_st,
    /* tmp: 1D memref */
    double *t_b, double *t_a, int64_t t_o, int64_t t_s, int64_t t_st);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_atax(int nx, int ny, double *A, double *x, double *y, double *tmp) {
  polygeist_cublas_time_begin();
  kernel_atax_impl(nx, ny,
                   A, A, 0, nx, ny, ny, 1,
                   x, x, 0, ny, 1,
                   y, y, 0, ny, 1,
                   tmp, tmp, 0, nx, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_atax nx=%d ny=%d  %.3f ms\n",
          nx, ny, ms);
}
