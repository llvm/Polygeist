#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#ifndef BINARY_OP
#define BINARY_OP 0
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)
extern void FUNCTION(float *, float *, float *);

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  float *a = (float *)malloc((size_t)N * sizeof(float));
  float *b = (float *)malloc((size_t)N * sizeof(float));
  float *out = (float *)malloc((size_t)N * sizeof(float));
  if (!a || !b || !out) return 2;
  for (int i = 0; i < N; ++i) {
    a[i] = (float)((i % 113) - 56) * 0.03125f;
    b[i] = (float)(((i * 7) % 109) - 54) * 0.025f;
    if (b[i] == 0.0f) b[i] = 0.125f;
  }
  FUNCTION(a, b, out);
  for (int i = 0; i < N; ++i) {
    float expected = BINARY_OP == 0 ? atan2f(a[i], b[i]) :
                     BINARY_OP == 1 ? remainderf(a[i], b[i]) :
                     BINARY_OP == 2 ? truncf(a[i] / b[i]) :
                     (b[i] > -3.0f && b[i] < 3.0f)
                         ? a[i] * 0.166666672f : 0.0f;
    if (fabsf(out[i] - expected) > 2.0e-5f) {
      fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, out[i], expected);
      return 1;
    }
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) FUNCTION(a, b, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), N, us);
  free(out); free(b); free(a);
  return 0;
}
