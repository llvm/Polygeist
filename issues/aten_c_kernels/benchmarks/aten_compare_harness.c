#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#ifndef COMPARE_OP
#define COMPARE_OP 0
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)
extern void FUNCTION(float *, float *, float *);

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static float expected(float a, float b) {
  int value = COMPARE_OP == 0 ? a == b : COMPARE_OP == 1 ? a != b :
              COMPARE_OP == 2 ? a < b  : COMPARE_OP == 3 ? a <= b :
              COMPARE_OP == 4 ? a > b  : COMPARE_OP == 5 ? a >= b :
              COMPARE_OP == 6 ? (a != 0.0f && b != 0.0f) :
              COMPARE_OP == 7 ? (a != 0.0f || b != 0.0f) :
                                ((a != 0.0f) != (b != 0.0f));
  return value ? 1.0f : 0.0f;
}

int main(void) {
  float *a = (float *)malloc((size_t)N * sizeof(float));
  float *b = (float *)malloc((size_t)N * sizeof(float));
  float *out = (float *)malloc((size_t)N * sizeof(float));
  if (!a || !b || !out) return 2;
  for (int i = 0; i < N; ++i) {
    a[i] = (float)((i % 113) - 56) * 0.03125f;
    b[i] = i % 17 == 0 ? a[i] :
           (float)(((i * 7) % 109) - 54) * 0.025f;
  }
  if (N > 4) {
    a[1] = NAN; b[1] = 1.0f;
    a[2] = 1.0f; b[2] = NAN;
    a[3] = NAN; b[3] = NAN;
  }
  FUNCTION(a, b, out);
  for (int i = 0; i < N; ++i) {
    float want = expected(a[i], b[i]);
    if (out[i] != want) {
      fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, out[i], want);
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
