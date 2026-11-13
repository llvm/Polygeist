#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void aten_nested_batch_offsets_cpu(int *, int *);

static double now_us(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return 1.0e6 * time.tv_sec + 1.0e-3 * time.tv_nsec;
}

int main(void) {
  int *size = malloc((size_t)B * sizeof(int));
  int *output = malloc((size_t)(B + 1) * sizeof(int));
  if (!size || !output) return 2;
  for (int i = 0; i < B; ++i) size[i] = (i % 7) + 1;
  aten_nested_batch_offsets_cpu(size, output);
  int expected = 0;
  for (int i = 0; i <= B; ++i) {
    if (output[i] != expected) {
      fprintf(stderr, "FAIL i=%d got=%d expected=%d\n", i, output[i], expected);
      return 1;
    }
    if (i < B) expected += size[i];
  }
  double begin = now_us();
  for (int i = 0; i < 5; ++i) aten_nested_batch_offsets_cpu(size, output);
  printf("RESULT kernel=aten_nested_batch_offsets_cpu B=%d warm_us=%.6f correctness=PASS\n",
         B, (now_us() - begin) / 5.0);
  free(output); free(size); return 0;
}
