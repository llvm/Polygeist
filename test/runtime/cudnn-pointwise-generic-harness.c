#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef POINTWISE_N
#define POINTWISE_N 4194304
#endif

void pointwise_generic(float *, float *, float, float, float *);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec / 1.0e6;
}

int main(void) {
  size_t bytes = (size_t)POINTWISE_N * sizeof(float);
  float *x = malloc(bytes), *y = malloc(bytes), *out = malloc(bytes);
  if (!x || !y || !out) return 2;
  const float scale = 0.75f, offset = -0.125f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    x[i] = (float)((i % 1009) - 504) / 256.0f;
    y[i] = (float)((i * 7 % 997) - 498) / 512.0f;
  }
  pointwise_generic(x, y, -0.5f, 0.25f, out);
  float binding_error = 0.0f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    float expected = tanhf((x[i] - y[i]) * -0.5f) + 0.25f;
    float error = fabsf(out[i] - expected);
    if (error > binding_error) binding_error = error;
  }
  for (int i = 0; i < 5; ++i)
    pointwise_generic(x, y, scale, offset, out);
  double start = now_ms();
  for (int i = 0; i < 20; ++i)
    pointwise_generic(x, y, scale, offset, out);
  double median_like_ms = (now_ms() - start) / 20.0;
  float max_error = 0.0f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    float expected = tanhf((x[i] - y[i]) * scale) + offset;
    float error = fabsf(out[i] - expected);
    if (error > max_error) max_error = error;
  }
  printf("generic_cudnn_graph N=%d nodes=4 average_warm_ms=%.6f "
         "max_abs_error=%g binding_error=%g correctness=%s\n",
         POINTWISE_N, median_like_ms, (double)max_error,
         (double)binding_error,
         max_error <= 1.0e-5f && binding_error <= 1.0e-5f
             ? "PASS" : "FAIL");
  free(out); free(y); free(x);
  return max_error <= 1.0e-5f && binding_error <= 1.0e-5f ? 0 : 1;
}
