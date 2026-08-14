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

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

#if defined(BENCH_REDUCE_F32) || defined(BENCH_TRACE_F32)
extern void FUNCTION(float *, float *);
static void call(float *x, float *out) { FUNCTION(x, out); }
#elif defined(BENCH_REDUCE_F64)
extern void FUNCTION(double *, double *);
static void call(double *x, double *out) { FUNCTION(x, out); }
#elif defined(BENCH_MINMAX_F32)
extern void FUNCTION(float *, float *, float *);
#elif !defined(BENCH_TRACE_F32)
#error "select BENCH_REDUCE_F32, BENCH_REDUCE_F64, or BENCH_MINMAX_F32"
#endif

int main(void) {
#if defined(BENCH_REDUCE_F64)
  double *x = (double *)malloc((size_t)N * sizeof(double));
  if (!x) return 2;
  double expected = 0.0, out = 0.0;
  for (int i = 0; i < N; ++i) {
    x[i] = ((double)((i * 17 + 5) % 101) - 50.0) / 257.0;
    expected += x[i];
  }
  expected /= (double)N;
  call(x, &out);
  if (fabs(out - expected) > 1.0e-10) {
    fprintf(stderr, "FAIL got=%.17g expected=%.17g\n", out, expected);
    return 1;
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) call(x, &out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
#elif defined(BENCH_MINMAX_F32)
  float *x = (float *)malloc((size_t)N * sizeof(float));
  if (!x) return 2;
  float expected_min, expected_max, out_min = 0.0f, out_max = 0.0f;
  for (int i = 0; i < N; ++i)
    x[i] = (float)(((i * 17 + 5) % 101) - 50) / 257.0f;
  expected_min = expected_max = x[0];
  for (int i = 1; i < N; ++i) {
    expected_min = x[i] < expected_min ? x[i] : expected_min;
    expected_max = x[i] > expected_max ? x[i] : expected_max;
  }
  FUNCTION(x, &out_min, &out_max);
  if (out_min != expected_min || out_max != expected_max) {
    fprintf(stderr, "FAIL min=%g/%g max=%g/%g\n", out_min, expected_min,
            out_max, expected_max);
    return 1;
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i)
    FUNCTION(x, &out_min, &out_max);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
#elif defined(BENCH_TRACE_F32)
  float *x = (float *)malloc((size_t)N * N * sizeof(float));
  if (!x) return 2;
  float expected = 0.0f, out = 0.0f;
  for (size_t i = 0; i < (size_t)N * N; ++i)
    x[i] = (float)((int)((i * 17 + 5) % 101) - 50) / 257.0f;
  for (int i = 0; i < N; ++i) expected += x[(size_t)i * N + i];
  call(x, &out);
  if (fabsf(out - expected) > 2.0e-4f) {
    fprintf(stderr, "FAIL got=%g expected=%g\n", out, expected);
    return 1;
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) call(x, &out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
#else
  float *x = (float *)malloc((size_t)N * sizeof(float));
  if (!x) return 2;
  float expected, out = 0.0f;
  for (int i = 0; i < N; ++i)
    x[i] = (float)(((i * 17 + 5) % 101) - 50) / 257.0f;
#if REDUCE_OP == 0
  expected = 0.0f;
  for (int i = 0; i < N; ++i) expected += x[i];
#elif REDUCE_OP == 1
  expected = 1.0f;
  for (int i = 0; i < N; ++i) expected *= x[i];
#elif REDUCE_OP == 2
  expected = x[0];
  for (int i = 1; i < N; ++i) expected = x[i] < expected ? x[i] : expected;
#elif REDUCE_OP == 3
  expected = x[0];
  for (int i = 1; i < N; ++i) expected = x[i] > expected ? x[i] : expected;
#else
#error "REDUCE_OP must be 0=sum, 1=product, 2=min, or 3=max"
#endif
  call(x, &out);
  // Parallel library reductions reassociate additions.  For very large,
  // cancellation-heavy vectors compare with an error proportional to the
  // accumulated rounding budget rather than requiring scalar loop order.
  float tolerance = REDUCE_OP == 0 ? fmaxf(2.0e-4f, 1.0e-9f * N)
                                   : 2.0e-4f;
  if (!(fabsf(out - expected) <= tolerance ||
        fabsf(out - expected) <= tolerance * fabsf(expected))) {
    fprintf(stderr, "FAIL got=%g expected=%g\n", out, expected);
    return 1;
  }
  double begin = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) call(x, &out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
#endif
  printf("RESULT function=%s N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), N, us);
  free(x);
  return 0;
}
