#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef POINTWISE_N
#define POINTWISE_N 4194304
#endif

#define WARMUP_RUNS 5
#define TIMED_RUNS 20

void pointwise_affine_relu(float *, float *, float, float *);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int compare_double(const void *a, const void *b) {
  double lhs = *(const double *)a;
  double rhs = *(const double *)b;
  return (lhs > rhs) - (lhs < rhs);
}

int main(void) {
  const size_t bytes = (size_t)POINTWISE_N * sizeof(float);
  float *x = (float *)malloc(bytes);
  float *bias = (float *)malloc(bytes);
  float *out = (float *)malloc(bytes);
  float *reference = (float *)malloc(bytes);
  if (!x || !bias || !out || !reference) return 2;

  const float alpha = 1.75f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    x[i] = (float)((i % 4093) - 2046) / 1024.0f;
    bias[i] = (float)((i * 17 % 1021) - 510) / 2048.0f;
    float value = alpha * x[i] + bias[i];
    reference[i] = value > 0.0f ? value : 0.0f;
  }

  // Exercise the same finalized graph with a different runtime scalar.  This
  // catches accidentally baking the matcher's semantic capture into a plan.
  const float alternate_alpha = -0.5f;
  pointwise_affine_relu(x, bias, alternate_alpha, out);
  float binding_error = 0.0f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    float value = alternate_alpha * x[i] + bias[i];
    float expected = value > 0.0f ? value : 0.0f;
    float error = fabsf(out[i] - expected);
    if (error > binding_error) binding_error = error;
  }

  for (int i = 0; i < WARMUP_RUNS; ++i)
    pointwise_affine_relu(x, bias, alpha, out);

  double samples[TIMED_RUNS];
  for (int i = 0; i < TIMED_RUNS; ++i) {
    double start = now_ms();
    pointwise_affine_relu(x, bias, alpha, out);
    samples[i] = now_ms() - start;
  }

  float max_abs_error = 0.0f;
  for (int i = 0; i < POINTWISE_N; ++i) {
    float error = fabsf(out[i] - reference[i]);
    if (error > max_abs_error) max_abs_error = error;
  }
  qsort(samples, TIMED_RUNS, sizeof(samples[0]), compare_double);
  double median = (samples[TIMED_RUNS / 2 - 1] +
                   samples[TIMED_RUNS / 2]) * 0.5;
  printf("backend=cudnn_graph N=%d alpha=%.2f warmups=%d runs=%d "
         "median_ms=%.6f max_abs_error=%g binding_error=%g correctness=%s\n",
         POINTWISE_N, alpha, WARMUP_RUNS, TIMED_RUNS, median,
         (double)max_abs_error, (double)binding_error,
         max_abs_error <= 1.0e-5f && binding_error <= 1.0e-5f
             ? "PASS" : "FAIL");

  free(reference);
  free(out);
  free(bias);
  free(x);
  return max_abs_error <= 1.0e-5f && binding_error <= 1.0e-5f ? 0 : 1;
}
