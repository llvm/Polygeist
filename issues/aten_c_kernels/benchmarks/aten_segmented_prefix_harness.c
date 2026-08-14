#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

#if PREFIX_KIND == 0
extern void FUNCTION(float *, int32_t *, float *);
#else
extern void FUNCTION(int32_t *, int32_t *, int32_t *);
#endif

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  size_t count = (size_t)B * N;
  int32_t *lengths = (int32_t *)malloc((size_t)B * sizeof(int32_t));
  if (!lengths) return 2;
  for (int32_t row = 0; row < B; ++row)
    lengths[row] = (int32_t)(((uint32_t)row * 37u + 11u) % (N + 1u));
#if PREFIX_KIND == 0
  float *x = (float *)malloc(count * sizeof(float));
  float *out = (float *)malloc((size_t)B * sizeof(float));
  float *expected = (float *)malloc((size_t)B * sizeof(float));
  if (!x || !out || !expected) return 2;
  for (size_t i = 0; i < count; ++i)
    x[i] = (float)((int32_t)(i % 19) - 9) * 0.0625f;
  for (int32_t row = 0; row < B; ++row) {
    float acc = 0.0f;
    for (int32_t col = 0; col < lengths[row]; ++col)
      acc += x[(size_t)row * N + col];
    expected[row] = acc;
  }
#else
  int32_t *x = (int32_t *)malloc(count * sizeof(int32_t));
  int32_t *out = (int32_t *)malloc((size_t)B * sizeof(int32_t));
  int32_t *expected = (int32_t *)malloc((size_t)B * sizeof(int32_t));
  if (!x || !out || !expected) return 2;
  for (size_t i = 0; i < count; ++i)
    x[i] = (i % 97 == 0) ? 0 : (int32_t)((i % 13) + 1);
  for (int32_t row = 0; row < B; ++row) {
    int32_t acc = 1;
    for (int32_t col = 0; col < lengths[row]; ++col)
      acc = acc != 0 && x[(size_t)row * N + col] != 0;
    expected[row] = acc;
  }
#endif
  FUNCTION(x, lengths, out);
  for (int32_t row = 0; row < B; ++row) {
#if PREFIX_KIND == 0
    if (fabsf(out[row] - expected[row]) > 1.0e-5f) {
      fprintf(stderr, "FAIL row=%d got=%g expected=%g\n",
              row, out[row], expected[row]);
#else
    if (out[row] != expected[row]) {
      fprintf(stderr, "FAIL row=%d got=%d expected=%d\n",
              row, out[row], expected[row]);
#endif
      return 1;
    }
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) FUNCTION(x, lengths, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s B=%d N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), B, N, us);
  free(expected); free(out); free(x); free(lengths);
  return 0;
}
