/* whisper_ops.c -- standalone Whisper/ggml-style operation fixtures.
 *
 * These source-level kernels isolate the compute shapes we want the linalg
 * raising pipeline to see from Whisper inference: dot product, softmax,
 * RMSNorm-like normalization, GELU, and the encoder-side 1D convolution.
 */

#include <math.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef N
#define N 128
#endif

#ifndef CONV_IN
#define CONV_IN 160
#endif

#ifndef CONV_K
#define CONV_K 3
#endif

#define CONV_OUT (CONV_IN - CONV_K + 1)
#define NEG_INF ((DATA_TYPE)-3.4028234663852886e38f)

void kernel_whisper_vec_dot(DATA_TYPE out[1], DATA_TYPE x[N],
                            DATA_TYPE y[N]) {
  DATA_TYPE sum = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < N; ++i) {
    sum += x[i] * y[i];
  }
  out[0] = sum;
#pragma endscop
}

DATA_TYPE kernel_whisper_vec_softmax(DATA_TYPE out[N], DATA_TYPE x[N],
                                     DATA_TYPE max_val) {
  DATA_TYPE sum = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < N; ++i) {
    DATA_TYPE val = expf(x[i] - max_val);
    out[i] = val;
    sum += val;
  }
#pragma endscop

  return sum;
}

void kernel_whisper_softmax_full(DATA_TYPE out[N], DATA_TYPE x[N]) {
  DATA_TYPE max_val = NEG_INF;

#pragma scop
  for (int i = 0; i < N; ++i) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  DATA_TYPE sum = (DATA_TYPE)0;
  for (int i = 0; i < N; ++i) {
    DATA_TYPE val = expf(x[i] - max_val);
    out[i] = val;
    sum += val;
  }

  DATA_TYPE inv_sum = (DATA_TYPE)1 / sum;
  for (int i = 0; i < N; ++i) {
    out[i] *= inv_sum;
  }
#pragma endscop
}

void kernel_whisper_rms_norm(DATA_TYPE out[N], DATA_TYPE x[N],
                             DATA_TYPE eps) {
  DATA_TYPE ss = (DATA_TYPE)0;

#pragma scop
  for (int i = 0; i < N; ++i) {
    ss += x[i] * x[i];
  }

  DATA_TYPE scale = (DATA_TYPE)1 / sqrtf(ss / (DATA_TYPE)N + eps);
  for (int i = 0; i < N; ++i) {
    out[i] = x[i] * scale;
  }
#pragma endscop
}

void kernel_whisper_gelu(DATA_TYPE out[N], DATA_TYPE x[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    DATA_TYPE v = x[i];
    DATA_TYPE inner = (DATA_TYPE)0.7978845608028654f *
                      (v + (DATA_TYPE)0.044715f * v * v * v);
    out[i] = (DATA_TYPE)0.5f * v * ((DATA_TYPE)1 + tanhf(inner));
  }
#pragma endscop
}

void kernel_whisper_conv1d(int n, int k, float *out, const float *x,
                           const float *filter) {
#pragma scop
  for (int i = 0; i <= n - k; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
      sum += x[i + j] * filter[j];
    }
    out[i] = sum;
  }
#pragma endscop
}
