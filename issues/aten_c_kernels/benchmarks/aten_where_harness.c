#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
extern void FUNCTION(int32_t *, float *, float *, float *);
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  int32_t *condition = malloc((size_t)N * sizeof(int32_t));
  float *x = malloc((size_t)N * sizeof(float));
  float *y = malloc((size_t)N * sizeof(float));
  float *out = malloc((size_t)N * sizeof(float));
  if (!condition || !x || !y || !out) return 2;
  for (int i = 0; i < N; ++i) {
    condition[i] = i % 3 ? 0 : -7;
    x[i] = (float)i * 0.25f;
    y[i] = (float)-i * 0.5f;
  }
  FUNCTION(condition, x, y, out);
  for (int i = 0; i < N; ++i) {
    float want = condition[i] ? x[i] : y[i];
    if (out[i] != want) {
      fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, out[i], want);
      return 1;
    }
  }
  double begin = seconds();
  for (int it = 0; it < BENCH_ITERS; ++it) FUNCTION(condition, x, y, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), N, us);
  free(out); free(y); free(x); free(condition);
  return 0;
}
