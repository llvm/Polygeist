#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
extern void FUNCTION(float *, int *, float *);
#define STRINGIFY1(x) #x
#define STRINGIFY(x) STRINGIFY1(x)
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
  float *input = malloc((size_t)N * sizeof(float));
  float *out = malloc((size_t)N * sizeof(float));
  int *index = malloc((size_t)N * sizeof(int));
  if (!input || !out || !index) return 2;
  for (int i = 0; i < N; ++i) { input[i] = i * 0.25f; index[i] = i % 13 == 0 ? -1 : (i * 17) % N; }
  FUNCTION(input, index, out);
  for (int i = 0; i < N; ++i) {
    float want = index[i] >= 0 ? input[index[i]] : 0.0f;
    if (out[i] != want) { fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, out[i], want); return 1; }
  }
  double begin = seconds();
  for (int it = 0; it < BENCH_ITERS; ++it) FUNCTION(input, index, out);
  printf("RESULT function=%s output_elements=%d warm_us=%.6f correctness=PASS\n", STRINGIFY(FUNCTION), N, (seconds()-begin)*1e6/BENCH_ITERS);
  free(index); free(out); free(input); return 0;
}
