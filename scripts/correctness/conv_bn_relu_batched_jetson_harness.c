/* conv_bn_relu_batched_jetson_harness.c — Jetson harness for the fused
 * conv + bn (inference) + relu pattern. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(LARGE_DATASET)
# define B  32
# define IC 64
# define OC 64
# define H  56
# define W  56
# define KS 3
#elif defined(MINI_DATASET)
# define B  4
# define IC 8
# define OC 8
# define H  32
# define W  32
# define KS 3
#endif
#ifndef B
# define B 4
#endif
#ifndef IC
# define IC 8
#endif
#ifndef OC
# define OC 8
#endif
#ifndef H
# define H 32
#endif
#ifndef W
# define W 32
#endif
#ifndef KS
# define KS 3
#endif
#define OH (H - KS + 1)
#define OW (W - KS + 1)
#define EPS 1e-5f

extern void kernel_conv_bn_relu_batched_impl(
    float *A_b,  float *A_a,  int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_s2, int64_t A_s3,
    int64_t A_t0, int64_t A_t1, int64_t A_t2, int64_t A_t3,
    float *F_b,  float *F_a,  int64_t F_o,
    int64_t F_s0, int64_t F_s1, int64_t F_s2, int64_t F_s3,
    int64_t F_t0, int64_t F_t1, int64_t F_t2, int64_t F_t3,
    float *S_b,  float *S_a,  int64_t S_o, int64_t S_sz, int64_t S_st,
    float *M_b,  float *M_a,  int64_t M_o, int64_t M_sz, int64_t M_st,
    float *I_b,  float *I_a,  int64_t I_o, int64_t I_sz, int64_t I_st,
    float *Bi_b, float *Bi_a, int64_t Bi_o, int64_t Bi_sz, int64_t Bi_st,
    float *O_b,  float *O_a,  int64_t O_o,
    int64_t O_s0, int64_t O_s1, int64_t O_s2, int64_t O_s3,
    int64_t O_t0, int64_t O_t1, int64_t O_t2, int64_t O_t3);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *F, float *scale, float *mean,
                        float *invst, float *bias, float *Bout) {
  polygeist_cublas_time_begin();
  kernel_conv_bn_relu_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)IC, (int64_t)H, (int64_t)W,
      (int64_t)(IC*H*W), (int64_t)(H*W), (int64_t)W, 1,
      F, F, 0,
      (int64_t)OC, (int64_t)IC, (int64_t)KS, (int64_t)KS,
      (int64_t)(IC*KS*KS), (int64_t)(KS*KS), (int64_t)KS, 1,
      scale, scale, 0, (int64_t)OC, 1,
      mean,  mean,  0, (int64_t)OC, 1,
      invst, invst, 0, (int64_t)OC, 1,
      bias,  bias,  0, (int64_t)OC, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)OC, (int64_t)OH, (int64_t)OW,
      (int64_t)(OC*OH*OW), (int64_t)(OH*OW), (int64_t)OW, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: conv_bn_relu_batched B=%d IC=%d OC=%d "
      "H=%d W=%d K=%d  %.3f ms\n",
      B, IC, OC, H, W, KS, ms);
}

int main(void) {
  size_t nA = (size_t)B*IC*H*W,
         nF = (size_t)OC*IC*KS*KS,
         nO = (size_t)B*OC*OH*OW;
  float *A     = (float *)malloc(nA * sizeof(float));
  float *F     = (float *)malloc(nF * sizeof(float));
  float *O     = (float *)malloc(nO * sizeof(float));
  float *scale = (float *)malloc(OC * sizeof(float));
  float *mean  = (float *)malloc(OC * sizeof(float));
  float *invst = (float *)malloc(OC * sizeof(float));
  float *bias  = (float *)malloc(OC * sizeof(float));
  if (!A || !F || !O || !scale || !mean || !invst || !bias) {
    fprintf(stderr, "alloc failed\n"); return 1;
  }

  for (int b = 0; b < B; ++b)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          A[((size_t)b*IC + c)*H*W + (size_t)i*W + j] =
              (float)((b + c + i + j) % 17) / 17.0f - 0.5f;  /* zero-mean-ish */
  for (int oc = 0; oc < OC; ++oc)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < KS; ++i)
        for (int j = 0; j < KS; ++j)
          F[((size_t)oc*IC + c)*KS*KS + (size_t)i*KS + j] =
              ((float)((oc*3 + c*5 + i*7 + j) % 11) / 11.0f) - 0.5f;
  for (int oc = 0; oc < OC; ++oc) {
    scale[oc] = 0.5f + 0.1f * (float)oc;
    mean[oc]  = 0.05f * (float)oc;
    float var = 0.2f + 0.01f * (float)oc;
    invst[oc] = 1.0f / sqrtf(var + EPS);
    bias[oc]  = 0.01f * (float)oc;
  }
  memset(O, 0, nO * sizeof(float));

  run_kernel(A, F, scale, mean, invst, bias, O);

  double sum = 0;
  size_t n_zero = 0;  /* relu activations that pinned to 0 */
  for (size_t k = 0; k < nO; ++k) {
    sum += O[k];
    if (O[k] == 0.0f) n_zero++;
  }
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems, %zu zeroed by ReLU (%.1f%%)\n",
          sum, nO, n_zero, 100.0 * (double)n_zero / (double)nO);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nO; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", O[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(F); free(O); free(scale); free(mean); free(invst); free(bias);
  return 0;
}
