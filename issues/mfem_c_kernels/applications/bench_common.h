#ifndef POLYGEIST_MFEM_BENCH_COMMON_H
#define POLYGEIST_MFEM_BENCH_COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 100
#endif

static unsigned bench_state = 0x6d2b79f5u;

static double bench_value(void) {
  bench_state = 1664525u * bench_state + 1013904223u;
  return ((double)((bench_state >> 8) & 0xffffu) / 32768.0) - 1.0;
}

static void bench_fill(double *x, int n) {
  for (int i = 0; i < n; ++i) x[i] = bench_value();
}

static void bench_transpose(const double *a, double *at, int rows, int cols) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) at[j * rows + i] = a[i * cols + j];
}

static double bench_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static double bench_max_error(const double *a, const double *b, int n) {
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = fabs(a[i] - b[i]);
    if (d > error) error = d;
  }
  return error;
}

static double bench_checksum(const double *x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += x[i] * (double)(i + 1);
  return sum;
}

#define RUN_AND_REPORT(APP, OP, CALL_REF, CALL_RAISED, YN)                    \
  do {                                                                        \
    memset(y_ref, 0, sizeof(y_ref));                                          \
    memset(y_raised, 0, sizeof(y_raised));                                    \
    CALL_REF;                                                                 \
    CALL_RAISED;                                                              \
    const double error = bench_max_error(y_ref, y_raised, (YN));              \
    memset(y_ref, 0, sizeof(y_ref));                                          \
    double begin = bench_now();                                               \
    for (int iteration = 0; iteration < BENCH_ITERS; ++iteration) CALL_REF;   \
    const double reference_seconds = bench_now() - begin;                     \
    memset(y_raised, 0, sizeof(y_raised));                                    \
    begin = bench_now();                                                      \
    for (int iteration = 0; iteration < BENCH_ITERS; ++iteration) CALL_RAISED;\
    const double raised_seconds = bench_now() - begin;                        \
    printf("app=%s operator=%s iterations=%d max_error=%.17g "                \
           "reference_us=%.6f raised_us=%.6f speedup=%.6f checksum=%.17g\n", \
           (APP), (OP), BENCH_ITERS, error,                                   \
           1.0e6 * reference_seconds / BENCH_ITERS,                           \
           1.0e6 * raised_seconds / BENCH_ITERS,                              \
           reference_seconds / raised_seconds,                               \
           bench_checksum(y_raised, (YN)));                                  \
    return error <= 5.0e-13 ? 0 : 1;                                         \
  } while (0)

#endif
