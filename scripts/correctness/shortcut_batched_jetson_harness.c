/* shortcut_batched_jetson_harness.c — Jetson harness for batched
 * residual-add shortcut. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define B 32
# define C 64
# define H 56
# define W 56
#elif defined(MINI_DATASET)
# define B 4
# define C 8
# define H 32
# define W 32
#endif
#ifndef B
# define B 4
#endif
#ifndef C
# define C 8
#endif
#ifndef H
# define H 32
#endif
#ifndef W
# define W 32
#endif

extern void kernel_shortcut_batched_impl(
    float *A_b, float *A_a, int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_s2, int64_t A_s3,
    int64_t A_t0, int64_t A_t1, int64_t A_t2, int64_t A_t3,
    float *O_b, float *O_a, int64_t O_o,
    int64_t O_s0, int64_t O_s1, int64_t O_s2, int64_t O_s3,
    int64_t O_t0, int64_t O_t1, int64_t O_t2, int64_t O_t3);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *Bout) {
  polygeist_cublas_time_begin();
  kernel_shortcut_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W,
      (int64_t)(C*H*W), (int64_t)(H*W), (int64_t)W, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W,
      (int64_t)(C*H*W), (int64_t)(H*W), (int64_t)W, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: shortcut_batched B=%d C=%d H=%d W=%d  %.3f ms\n",
      B, C, H, W, ms);
}

int main(void) {
  size_t n = (size_t)B*C*H*W;
  float *A = (float *)malloc(n * sizeof(float));
  float *Bout = (float *)malloc(n * sizeof(float));
  if (!A || !Bout) { fprintf(stderr, "alloc failed\n"); return 1; }

  for (size_t k = 0; k < n; ++k) {
    A[k] = (float)((k * 17) % 41) / 41.0f;
    Bout[k] = (float)((k * 23) % 37) / 37.0f;
  }

  run_kernel(A, Bout);

  double sum = 0;
  for (size_t k = 0; k < n; ++k) sum += Bout[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, n);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < n; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", Bout[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(Bout);
  return 0;
}
