/* maxpool_batched_jetson_harness.c — Jetson harness for batched maxpool. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define B   32
# define C   64
# define H   112
# define W   112
# define KS  3
# define STR 2
#elif defined(MINI_DATASET)
# define B   4
# define C   8
# define H   32
# define W   32
# define KS  2
# define STR 2
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
#ifndef KS
# define KS 2
#endif
#ifndef STR
# define STR 2
#endif
#define OH ((H - KS) / STR + 1)
#define OW ((W - KS) / STR + 1)

extern void kernel_maxpool_batched_impl(
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
  kernel_maxpool_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W,
      (int64_t)(C*H*W), (int64_t)(H*W), (int64_t)W, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)C, (int64_t)OH, (int64_t)OW,
      (int64_t)(C*OH*OW), (int64_t)(OH*OW), (int64_t)OW, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: maxpool_batched B=%d C=%d H=%d W=%d K=%d S=%d  %.3f ms\n",
      B, C, H, W, KS, STR, ms);
}

int main(void) {
  size_t nA = (size_t)B*C*H*W, nO = (size_t)B*C*OH*OW;
  float *A = (float *)malloc(nA * sizeof(float));
  float *O = (float *)malloc(nO * sizeof(float));
  if (!A || !O) { fprintf(stderr, "alloc failed\n"); return 1; }

  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          A[((size_t)b*C + c)*H*W + (size_t)i*W + j] =
              (float)((b*7 + c*3 + i*5 + j*11) % 23) / 23.0f;
  memset(O, 0, nO * sizeof(float));

  run_kernel(A, O);

  double sum = 0;
  for (size_t k = 0; k < nO; ++k) sum += O[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, nO);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nO; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", O[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(O);
  return 0;
}
