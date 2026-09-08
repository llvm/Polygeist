#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 2000
#define WARMUPS 5
#define SAMPLES 5

extern void polygeist_cusolver_dpotrf_lower_row_major(int32_t n, double *a);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* This is the canonical PolyBench/C initializer, including accumulation order. */
static void initialize(double *a) {
  double *base = malloc((size_t)N * N * sizeof(double));
  if (!base) abort();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j)
      base[(size_t)i * N + j] = (double)(-j % N) / N + 1.0;
    for (int j = i + 1; j < N; ++j)
      base[(size_t)i * N + j] = 0.0;
    base[(size_t)i * N + i] = 1.0;
  }
  memset(a, 0, (size_t)N * N * sizeof(double));
  for (int t = 0; t < N; ++t)
    for (int r = 0; r < N; ++r)
      for (int s = 0; s < N; ++s)
        a[(size_t)r * N + s] +=
            base[(size_t)r * N + t] * base[(size_t)s * N + t];
  free(base);
}

int main(void) {
  const size_t bytes = (size_t)N * N * sizeof(double);
  double *original = malloc(bytes);
  double *work = malloc(bytes);
  if (!original || !work) return 2;
  initialize(original);
  for (int i = 0; i < WARMUPS + SAMPLES; ++i) {
    memcpy(work, original, bytes);
    const double start = now_ms();
    polygeist_cusolver_dpotrf_lower_row_major(N, work);
    const double elapsed = now_ms() - start;
    printf("CHOLESKY_SAMPLE phase=%s index=%d e2e_host_ms=%.6f\n",
           i < WARMUPS ? "warmup" : "sample",
           i < WARMUPS ? i + 1 : i - WARMUPS + 1, elapsed);
    fflush(stdout);
  }
  free(work);
  free(original);
  return 0;
}
