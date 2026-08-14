#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#ifndef I32_OP
#define I32_OP 0
#endif
#ifdef I32_UNARY
extern void FUNCTION(int32_t *, int32_t *);
#else
extern void FUNCTION(int32_t *, int32_t *, int32_t *);
#endif
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  int32_t *a = malloc((size_t)N * sizeof(int32_t));
  int32_t *b = malloc((size_t)N * sizeof(int32_t));
  int32_t *out = malloc((size_t)N * sizeof(int32_t));
  if (!a || !b || !out) return 2;
  for (int i = 0; i < N; ++i) {
    a[i] = (int32_t)(i * 1103515245u + 12345u);
    b[i] = I32_OP >= 3 ? (i % 31) : (int32_t)(i * 2654435761u);
  }
#ifdef I32_UNARY
  FUNCTION(a, out);
#else
  FUNCTION(a, b, out);
#endif
  for (int i = 0; i < N; ++i) {
    uint32_t shift = (uint32_t)b[i] & 31u;
    int32_t want = I32_OP == 0 ? (a[i] & b[i]) :
                   I32_OP == 1 ? (a[i] | b[i]) :
                   I32_OP == 2 ? (a[i] ^ b[i]) :
                   I32_OP == 3 ? (int32_t)((uint32_t)a[i] << shift) :
                   I32_OP == 4 ? (a[i] >> shift) : ~a[i];
    if (out[i] != want) {
      fprintf(stderr, "FAIL i=%d got=%d expected=%d\n", i, out[i], want);
      return 1;
    }
  }
  double begin = seconds();
  for (int it = 0; it < BENCH_ITERS; ++it) {
#ifdef I32_UNARY
    FUNCTION(a, out);
#else
    FUNCTION(a, b, out);
#endif
  }
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s N=%d warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), N, us);
  free(out); free(b); free(a);
  return 0;
}
