/* mvt_jetson_wrapper.c — Jetson timing wrapper.
 *
 * polybenchGpu kernel_mvt computes:
 *   x1 += A · y_1
 *   x2 += Aᵀ · y_2
 *
 * (Both are accumulating gemvs; the matcher fissions the accumulation,
 * so each surfaces as a plain gemv that writes to x1/x2 — initialised
 * elsewhere. The transpose-discriminator routes the second to dgemv_T.)
 *
 * Signature: kernel_mvt(n, x1, x2, y_1, y_2, A)
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_mvt_impl(
    int n,
    /* x1: 1D */
    double *x1_b, double *x1_a, int64_t x1_o, int64_t x1_s, int64_t x1_st,
    /* x2: 1D */
    double *x2_b, double *x2_a, int64_t x2_o, int64_t x2_s, int64_t x2_st,
    /* y_1: 1D */
    double *y1_b, double *y1_a, int64_t y1_o, int64_t y1_s, int64_t y1_st,
    /* y_2: 1D */
    double *y2_b, double *y2_a, int64_t y2_o, int64_t y2_s, int64_t y2_st,
    /* A: 2D */
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_mvt(int n, double *x1, double *x2, double *y_1, double *y_2,
                double *A) {
  polygeist_cublas_time_begin();
  kernel_mvt_impl(n,
                  x1, x1, 0, n, 1,
                  x2, x2, 0, n, 1,
                  y_1, y_1, 0, n, 1,
                  y_2, y_2, 0, n, 1,
                  A, A, 0, n, n, n, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_mvt n=%d  %.3f ms\n", n, ms);
}
