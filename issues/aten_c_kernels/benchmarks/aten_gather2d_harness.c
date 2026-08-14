#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
extern void FUNCTION(float *, int32_t *, float *);
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  size_t data_n = (size_t)R * S, out_n = (size_t)R * K;
  float *data = malloc(data_n * sizeof(float));
  int32_t *index = malloc(out_n * sizeof(int32_t));
  float *out = malloc(out_n * sizeof(float));
  if (!data || !index || !out) return 2;
  for (size_t i = 0; i < data_n; ++i) data[i] = (float)(i % 1009) * 0.125f;
  for (size_t i = 0; i < out_n; ++i) index[i] = (int32_t)((i * 17 + 3) % S);
  FUNCTION(data, index, out);
  for (int32_t r = 0; r < R; ++r)
    for (int32_t k = 0; k < K; ++k) {
      size_t p = (size_t)r * K + k;
      float want = data[(size_t)r * S + index[p]];
      if (out[p] != want) {
        fprintf(stderr, "FAIL p=%zu got=%g expected=%g\n", p, out[p], want);
        return 1;
      }
    }
  double begin = seconds();
  for (int it = 0; it < BENCH_ITERS; ++it) FUNCTION(data, index, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s R=%d K=%d S=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), R, K, S, us);
  free(out); free(index); free(data);
  return 0;
}
