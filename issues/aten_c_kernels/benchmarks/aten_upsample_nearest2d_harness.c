#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 10
#endif
#ifndef EXACT
#define EXACT 0
#endif
extern void FUNCTION(float *, float *);
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

int main(void) {
  size_t outer = (size_t)B * C;
  size_t input_n = outer * I0 * I1, output_n = outer * O0 * O1;
  float *input = malloc(input_n * sizeof(float));
  float *out = malloc(output_n * sizeof(float));
  if (!input || !out) return 2;
  for (size_t i = 0; i < input_n; ++i) input[i] = (float)(i % 1009) * 0.25f;
  FUNCTION(input, out);
  for (size_t q = 0; q < outer; ++q)
    for (int o0 = 0; o0 < O0; ++o0)
      for (int o1 = 0; o1 < O1; ++o1) {
        int i0 = EXACT ? ((2 * o0 + 1) * I0) / (2 * O0) : o0 * I0 / O0;
        int i1 = EXACT ? ((2 * o1 + 1) * I1) / (2 * O1) : o1 * I1 / O1;
        if (i0 >= I0) i0 = I0 - 1;
        if (i1 >= I1) i1 = I1 - 1;
        size_t p = (q * O0 + o0) * O1 + o1;
        float want = input[(q * I0 + i0) * I1 + i1];
        if (out[p] != want) {
          fprintf(stderr, "FAIL p=%zu got=%g expected=%g\n", p, out[p], want);
          return 1;
        }
      }
  double begin = seconds();
  for (int it = 0; it < BENCH_ITERS; ++it) FUNCTION(input, out);
  double us = (seconds() - begin) * 1.0e6 / BENCH_ITERS;
  printf("RESULT function=%s output_elements=%zu warm_us=%.6f correctness=PASS\n",
         STRINGIFY(FUNCTION), output_n, us);
  free(out); free(input);
  return 0;
}
