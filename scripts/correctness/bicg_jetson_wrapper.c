/* bicg_jetson_wrapper.c — Jetson timing wrapper.
 *
 * polybenchGpu kernel_bicg computes:
 *   s = Aᵀ·r   (gemv)
 *   q = A·p    (gemv)
 *
 * Bridges polybenchGpu's kernel_bicg(nx, ny, A, s, q, p, r) to the
 * MLIR-lowered kernel_bicg_impl with memref-descriptor args.
 */
#include <stdint.h>
#include <stdio.h>

extern void kernel_bicg_impl(
    int nx, int ny,
    /* A: 2D memref */
    double *A_b, double *A_a, int64_t A_o, int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    /* s: 1D memref */
    double *s_b, double *s_a, int64_t s_o, int64_t s_s, int64_t s_st,
    /* q: 1D memref */
    double *q_b, double *q_a, int64_t q_o, int64_t q_s, int64_t q_st,
    /* p: 1D memref */
    double *p_b, double *p_a, int64_t p_o, int64_t p_s, int64_t p_st,
    /* r: 1D memref */
    double *r_b, double *r_a, int64_t r_o, int64_t r_s, int64_t r_st);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

void kernel_bicg(int nx, int ny, double *A, double *s, double *q,
                 double *p, double *r) {
  polygeist_cublas_time_begin();
  kernel_bicg_impl(nx, ny,
                   A, A, 0, nx, ny, ny, 1,
                   s, s, 0, ny, 1,
                   q, q, 0, nx, 1,
                   p, p, 0, ny, 1,
                   r, r, 0, nx, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr, "POLYGEIST_TIMING: kernel_bicg nx=%d ny=%d  %.3f ms\n",
          nx, ny, ms);
}
