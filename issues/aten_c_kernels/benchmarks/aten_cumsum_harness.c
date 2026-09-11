#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void aten_cumsum(float *, float *);

static double now_us(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return 1.0e6 * time.tv_sec + 1.0e-3 * time.tv_nsec;
}

int main(void) {
  float *input = malloc((size_t)N * sizeof(float));
  float *output = malloc((size_t)N * sizeof(float));
  if (!input || !output) return 2;
  for (int i = 0; i < N; ++i) input[i] = (float)((i % 17) - 8) / 1024.0f;
  aten_cumsum(input, output);
  float expected = 0.0f;
  for (int i = 0; i < N; ++i) {
    expected += input[i];
    if (fabsf(output[i] - expected) > 2.0e-3f * (1.0f + fabsf(expected))) {
      fprintf(stderr, "FAIL i=%d got=%g expected=%g\n", i, output[i], expected);
      return 1;
    }
  }
  double begin = now_us();
  for (int i = 0; i < 5; ++i) aten_cumsum(input, output);
  printf("RESULT kernel=aten_cumsum N=%d warm_us=%.6f correctness=PASS\n",
         N, (now_us() - begin) / 5.0);
  free(output);
  free(input);
  return 0;
}
