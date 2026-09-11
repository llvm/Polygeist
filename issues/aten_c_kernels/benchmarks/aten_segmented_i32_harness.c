#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

extern void FUNCTION(int32_t *, int32_t *);

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  size_t count = (size_t)R * K;
  int32_t *x = (int32_t *)malloc(count * sizeof(int32_t));
  int32_t *out = (int32_t *)malloc((size_t)R * sizeof(int32_t));
  int32_t *expected = (int32_t *)malloc((size_t)R * sizeof(int32_t));
  if (!x || !out || !expected) return 2;
  for (size_t i = 0; i < count; ++i) {
    int32_t value = (int32_t)((i * 17 + 5) % 31) - 15;
    x[i] = i % 23 == 0 ? 0 : value;
  }
  for (int32_t row = 0; row < R; ++row) {
    int32_t acc = REDUCE_OP == 0 ? 1 : 0;
    for (int32_t col = 0; col < K; ++col) {
      int32_t value = x[(size_t)row * K + col];
      if (REDUCE_OP == 0) acc = acc != 0 && value != 0;
      else if (REDUCE_OP == 1) acc = acc != 0 || value != 0;
      else acc ^= value;
    }
    expected[row] = acc;
  }
  FUNCTION(x, out);
  for (int32_t row = 0; row < R; ++row)
    if (out[row] != expected[row]) {
      fprintf(stderr, "FAIL row=%d got=%d expected=%d\n",
              row, out[row], expected[row]);
      return 1;
    }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) FUNCTION(x, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s R=%d K=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), R, K, us);
  free(expected); free(out); free(x);
  return 0;
}
