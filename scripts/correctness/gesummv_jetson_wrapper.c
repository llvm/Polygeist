/* gesummv_jetson_wrapper.c — Jetson timing wrapper.
 *
 * gesummv: y = α·(A·x) + β·(B·x).
 * Signature: (n, α, β, A, B, tmp, x, y).
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_gesummv_impl(
    int n, double alpha, double beta,
    /* A: 2D */
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    /* B: 2D */
    double *B_b, double *B_a, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1,
    /* tmp,x,y: 1D each */
    double *tmp_b, double *tmp_a, int64_t tmp_o, int64_t tmp_s, int64_t tmp_st,
    double *x_b,   double *x_a,   int64_t x_o,   int64_t x_s,   int64_t x_st,
    double *y_b,   double *y_a,   int64_t y_o,   int64_t y_s,   int64_t y_st);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_gesummv(int n, double alpha, double beta, double *A, double *B,
                    double *tmp, double *x, double *y) {
  polygeist_cublas_time_begin();
  kernel_gesummv_impl(n, alpha, beta,
                      A, A, 0, n, n, n, 1,
                      B, B, 0, n, n, n, 1,
                      tmp, tmp, 0, n, 1,
                      x,   x,   0, n, 1,
                      y,   y,   0, n, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_gesummv n=%d  %.3f ms\n", n, ms);
}
