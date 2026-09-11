/* gemver_jetson_wrapper.c — Jetson timing wrapper.
 *
 * gemver: A = A + u1·v1ᵀ + u2·v2ᵀ; x = β·Aᵀ·y + z; w = α·A·x
 * Signature: (n, α, β, A, u1, v1, u2, v2, w, x, y, z).
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_gemver_impl(
    int n, double alpha, double beta,
    /* A: 2D */
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    /* u1,v1,u2,v2,w,x,y,z : 1D each (8 vectors) */
    double *u1_b, double *u1_a, int64_t u1_o, int64_t u1_s, int64_t u1_st,
    double *v1_b, double *v1_a, int64_t v1_o, int64_t v1_s, int64_t v1_st,
    double *u2_b, double *u2_a, int64_t u2_o, int64_t u2_s, int64_t u2_st,
    double *v2_b, double *v2_a, int64_t v2_o, int64_t v2_s, int64_t v2_st,
    double *w_b,  double *w_a,  int64_t w_o,  int64_t w_s,  int64_t w_st,
    double *x_b,  double *x_a,  int64_t x_o,  int64_t x_s,  int64_t x_st,
    double *y_b,  double *y_a,  int64_t y_o,  int64_t y_s,  int64_t y_st,
    double *z_b,  double *z_a,  int64_t z_o,  int64_t z_s,  int64_t z_st);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_gemver(int n, double alpha, double beta, double *A,
                   double *u1, double *v1, double *u2, double *v2,
                   double *w, double *x, double *y, double *z) {
  polygeist_cublas_time_begin();
  kernel_gemver_impl(n, alpha, beta,
                     A, A, 0, n, n, n, 1,
                     u1, u1, 0, n, 1,
                     v1, v1, 0, n, 1,
                     u2, u2, 0, n, 1,
                     v2, v2, 0, n, 1,
                     w,  w,  0, n, 1,
                     x,  x,  0, n, 1,
                     y,  y,  0, n, 1,
                     z,  z,  0, n, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_gemver n=%d  %.3f ms\n", n, ms);
}
