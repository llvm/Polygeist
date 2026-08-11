#include "polygeist_cublas_rt.h"
#include <math.h>
#include <stdio.h>

static int closef(float a, float b) { return fabsf(a - b) < 1.0e-5f; }

int main(void) {
  float a[6] = {1, 2, 3, 4, 5, 6};
  float b[6] = {7, 8, 9, 10, 11, 12};
  float c[4] = {1, 1, 1, 1};
  polygeist_cublas_sgemm_transpose(2, 2, 3, 0, 0, 1, a, 3, b, 2,
                                    0, c, 2);
  const float expect[4] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i)
    if (!closef(c[i], expect[i])) return 1;

  float x[3] = {1, 2, 3}, y[3] = {4, 5, 6};
  polygeist_cublas_saxpby(3, 2, x, 3, y);
  polygeist_cublas_sscal(3, 0.5f, y);
  const float vectorExpect[3] = {7, 9.5f, 12};
  for (int i = 0; i < 3; ++i)
    if (!closef(y[i], vectorExpect[i])) return 2;

  puts("cublas-f32-reference: PASS");
  return 0;
}
