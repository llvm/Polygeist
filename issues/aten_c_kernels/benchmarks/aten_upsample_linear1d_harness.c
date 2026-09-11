#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
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
  size_t input_n = outer * I0, output_n = outer * O0;
  float *input = malloc(input_n * sizeof(float));
  float *out = malloc(output_n * sizeof(float));
  if (!input || !out) return 2;
  for (size_t i = 0; i < input_n; ++i)
    input[i] = (float)(i % 1009) * 0.125f;
  FUNCTION(input, out);
  for (size_t q = 0; q < outer; ++q)
    for (int o = 0; o < O0; ++o) {
      float source = ((float)o + 0.5f) * (float)I0 / (float)O0 - 0.5f;
      if (source < 0.0f) source = 0.0f;
      int lower = (int)source;
      int upper = lower + 1 < I0 ? lower + 1 : lower;
      float wu = source - (float)lower;
      float want = input[q * I0 + lower] * (1.0f - wu) +
                   input[q * I0 + upper] * wu;
      float got = out[q * O0 + o];
      if (fabsf(got - want) > 2.0e-5f * fmaxf(1.0f, fabsf(want))) {
        fprintf(stderr, "FAIL q=%zu o=%d got=%g expected=%g\n",
                q, o, got, want);
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
