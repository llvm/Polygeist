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

  float matrix[6] = {1, 2, 3, 4, 5, 6};
  float input[3] = {1, 2, 3};
  float output[2] = {0, 0};
  polygeist_cublas_sgemv(2, 3, 1, matrix, 3, input, 0, output);
  if (!closef(output[0], 14) || !closef(output[1], 32)) return 3;

  // Exercise the x/y alias semantics implemented by both CUDA SGEMV shims.
  float square[4] = {1, 2, 3, 4};
  float inplace[2] = {5, 6};
  polygeist_cublas_sgemv(2, 2, 1, square, 2, inplace, 0, inplace);
  if (!closef(inplace[0], 17) || !closef(inplace[1], 39)) return 4;

  float transposeInput[2] = {5, 6};
  polygeist_cublas_sgemv_T(
      2, 2, 1, square, 2, transposeInput, 0, transposeInput);
  if (!closef(transposeInput[0], 23) ||
      !closef(transposeInput[1], 34)) return 5;

  puts("cublas-f32-reference: PASS");
  return 0;
}
