/* Jetson harness for 1×1 conv routed to batched cublasSgemm. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define B  32
# define IC 256
# define OC 256
# define H  56
# define W  56
#elif defined(MINI_DATASET)
# define B  4
# define IC 16
# define OC 16
# define H  32
# define W  32
#endif
#ifndef B
# define B 4
#endif
#ifndef IC
# define IC 16
#endif
#ifndef OC
# define OC 16
#endif
#ifndef H
# define H 32
#endif
#ifndef W
# define W 32
#endif
#define KS 1
#define OH H
#define OW W

extern void kernel_conv1x1_batched_impl(
    float *A_b, float *A_a, int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_s2, int64_t A_s3,
    int64_t A_t0, int64_t A_t1, int64_t A_t2, int64_t A_t3,
    float *F_b, float *F_a, int64_t F_o,
    int64_t F_s0, int64_t F_s1, int64_t F_s2, int64_t F_s3,
    int64_t F_t0, int64_t F_t1, int64_t F_t2, int64_t F_t3,
    float *O_b, float *O_a, int64_t O_o,
    int64_t O_s0, int64_t O_s1, int64_t O_s2, int64_t O_s3,
    int64_t O_t0, int64_t O_t1, int64_t O_t2, int64_t O_t3);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *F, float *Bout) {
  polygeist_cublas_time_begin();
  kernel_conv1x1_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)IC, (int64_t)H, (int64_t)W,
      (int64_t)(IC*H*W), (int64_t)(H*W), (int64_t)W, 1,
      F, F, 0,
      (int64_t)OC, (int64_t)IC, (int64_t)KS, (int64_t)KS,
      (int64_t)(IC*KS*KS), (int64_t)(KS*KS), (int64_t)KS, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)OC, (int64_t)OH, (int64_t)OW,
      (int64_t)(OC*OH*OW), (int64_t)(OH*OW), (int64_t)OW, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: conv1x1_batched B=%d IC=%d OC=%d H=%d W=%d  %.3f ms\n",
      B, IC, OC, H, W, ms);
}

int main(void) {
  size_t nA = (size_t)B*IC*H*W, nF = (size_t)OC*IC, nO = (size_t)B*OC*OH*OW;
  float *A = (float *)malloc(nA * sizeof(float));
  float *F = (float *)malloc(nF * sizeof(float));
  float *O = (float *)malloc(nO * sizeof(float));
  if (!A || !F || !O) { fprintf(stderr, "alloc failed\n"); return 1; }

  for (size_t k = 0; k < nA; ++k)
    A[k] = (float)((k * 17) % 31) / 31.0f - 0.5f;
  for (size_t k = 0; k < nF; ++k)
    F[k] = (float)((k * 23) % 37) / 37.0f - 0.5f;
  memset(O, 0, nO * sizeof(float));

  run_kernel(A, F, O);

  double sum = 0;
  for (size_t k = 0; k < nO; ++k) sum += O[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, nO);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nO; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", O[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(F); free(O);
  return 0;
}
