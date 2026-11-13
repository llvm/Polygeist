#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
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
  float *x = (float *)malloc((size_t)B * sizeof(float));
  float *out = (float *)malloc((size_t)B * N * sizeof(float));
  if (!x || !out) return 2;
  for (int i = 0; i < B; ++i) x[i] = (float)(i % 29) * 0.125f - 1.0f;
  FUNCTION(x, out);
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < N; ++j)
      if (out[(size_t)i * N + j] != x[i]) {
        fprintf(stderr, "FAIL i=%d j=%d got=%g expected=%g\n",
                i, j, out[(size_t)i * N + j], x[i]);
        return 1;
      }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) FUNCTION(x, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s B=%d N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), B, N, us);
  free(out); free(x);
  return 0;
}
