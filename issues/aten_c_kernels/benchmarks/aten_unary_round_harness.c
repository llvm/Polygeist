#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#ifndef UNARY_OP
#define UNARY_OP 0
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)
extern void FUNCTION(float *, float *);

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  float *x = (float *)malloc((size_t)N * sizeof(float));
  float *out = (float *)malloc((size_t)N * sizeof(float));
  if (!x || !out) return 2;
  for (int i = 0; i < N; ++i)
    x[i] = (float)((i % 257) - 128) * 0.125f + (i & 1 ? 0.5f : -0.5f);
  FUNCTION(x, out);
  for (int i = 0; i < N; ++i) {
    float want = UNARY_OP == 0 ? roundf(x[i]) : UNARY_OP == 1 ? truncf(x[i]) :
                 UNARY_OP == 2 ? (x[i] == 0.0f ? 1.0f : 0.0f) :
                 UNARY_OP == 3 ? (float)((0.0f < x[i]) - (x[i] < 0.0f)) :
                 UNARY_OP == 4 ? (x[i] < 0.0f ? 1.0f : 0.0f) :
                                 (x[i] < 0.0f ? 3.14159274f : 0.0f);
    if (!(out[i] == want || (isnan(out[i]) && isnan(want)))) {
      fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, out[i], want);
      return 1;
    }
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) FUNCTION(x, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), N, us);
  free(out); free(x);
  return 0;
}
