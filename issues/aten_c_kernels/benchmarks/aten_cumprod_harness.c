#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void aten_cumprod_cpu(float *, float *);

static double now_us(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return 1.0e6 * time.tv_sec + 1.0e-3 * time.tv_nsec;
}

int main(void) {
  size_t count = (size_t)R * K;
  float *input = malloc(count * sizeof(float));
  float *output = malloc(count * sizeof(float));
  if (!input || !output) return 2;
  for (size_t i = 0; i < count; ++i)
    input[i] = 0.999f + 0.00002f * (float)(i % 101);
  aten_cumprod_cpu(input, output);
  for (int row = 0; row < R; ++row) {
    float expected = 1.0f;
    for (int col = 0; col < K; ++col) {
      size_t i = (size_t)row * K + col;
      expected *= input[i];
      if (fabsf(output[i] - expected) > 2.0e-4f * (1.0f + fabsf(expected))) {
        fprintf(stderr, "FAIL row=%d col=%d got=%g expected=%g\n",
                row, col, output[i], expected);
        return 1;
      }
    }
  }
  double begin = now_us();
  for (int i = 0; i < 5; ++i) aten_cumprod_cpu(input, output);
  printf("RESULT kernel=aten_cumprod_cpu R=%d K=%d warm_us=%.6f correctness=PASS\n",
         R, K, (now_us() - begin) / 5.0);
  free(output);
  free(input);
  return 0;
}
