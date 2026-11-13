/* batchnorm_batched_jetson_harness.c — Jetson harness for batched
 * per-channel batchnorm (inference). */
#include <math.h>
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
#define EPS 1e-5f

extern void kernel_batchnorm_batched_impl(
    float *A_b,     float *A_a,     int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_s2, int64_t A_s3,
    int64_t A_t0, int64_t A_t1, int64_t A_t2, int64_t A_t3,
    float *S_b,     float *S_a,     int64_t S_o, int64_t S_sz, int64_t S_st,
    float *M_b,     float *M_a,     int64_t M_o, int64_t M_sz, int64_t M_st,
    float *I_b,     float *I_a,     int64_t I_o, int64_t I_sz, int64_t I_st,
    float *Bi_b,    float *Bi_a,    int64_t Bi_o, int64_t Bi_sz, int64_t Bi_st,
    float *O_b,     float *O_a,     int64_t O_o,
    int64_t O_s0, int64_t O_s1, int64_t O_s2, int64_t O_s3,
    int64_t O_t0, int64_t O_t1, int64_t O_t2, int64_t O_t3);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *scale, float *mean,
                        float *inv_std, float *bias, float *Bout) {
  polygeist_cublas_time_begin();
  kernel_batchnorm_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W,
      (int64_t)(C*H*W), (int64_t)(H*W), (int64_t)W, 1,
      scale,   scale,   0, (int64_t)C, 1,
      mean,    mean,    0, (int64_t)C, 1,
      inv_std, inv_std, 0, (int64_t)C, 1,
      bias,    bias,    0, (int64_t)C, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W,
      (int64_t)(C*H*W), (int64_t)(H*W), (int64_t)W, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: batchnorm_batched B=%d C=%d H=%d W=%d  %.3f ms\n",
      B, C, H, W, ms);
}

int main(void) {
  size_t nA = (size_t)B*C*H*W;
  float *A     = (float *)malloc(nA * sizeof(float));
  float *Bout  = (float *)malloc(nA * sizeof(float));
  float *scale = (float *)malloc(C * sizeof(float));
  float *mean  = (float *)malloc(C * sizeof(float));
  float *invst = (float *)malloc(C * sizeof(float));
  float *bias  = (float *)malloc(C * sizeof(float));
  if (!A || !Bout || !scale || !mean || !invst || !bias) {
    fprintf(stderr, "alloc failed\n"); return 1;
  }

  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          A[((size_t)b*C + c)*H*W + (size_t)i*W + j] =
              (float)((b*2 + c*3 + i*5 + j*7) % 29) / 29.0f;
  for (int c = 0; c < C; ++c) {
    scale[c] = 0.5f + 0.1f * (float)c;
    mean[c]  = 0.05f * (float)c;
    /* var ~ small positive; inv_std = 1/sqrt(var+eps) */
    float var = 0.2f + 0.01f * (float)c;
    invst[c] = 1.0f / sqrtf(var + EPS);
    bias[c]  = 0.01f * (float)c;
  }
  memset(Bout, 0, nA * sizeof(float));

  run_kernel(A, scale, mean, invst, bias, Bout);

  double sum = 0;
  for (size_t k = 0; k < nA; ++k) sum += Bout[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, nA);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nA; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", Bout[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(Bout); free(scale); free(mean); free(invst); free(bias);
  return 0;
}
