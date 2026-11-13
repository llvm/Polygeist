#include "polygeist_cublas_rt.h"
#include <math.h>
#include <stdio.h>

static int closef(float a, float b) { return fabsf(a - b) < 1.0e-5f; }
static int closed(double a, double b) { return fabs(a - b) < 1.0e-12; }

int main(void) {
  float x[4] = {2.0f, -3.0f, 4.0f, 5.0f};
  float sum = 7.0f;
  polygeist_cudnn_reduce_f32(0, 4, x, &sum);
  if (!closef(sum, 15.0f)) return 1;
  float product = 2.0f;
  polygeist_cudnn_reduce_f32(1, 4, x, &product);
  if (!closef(product, -240.0f)) return 2;
  float minimum = 1.0f;
  polygeist_cudnn_reduce_f32(2, 4, x, &minimum);
  if (!closef(minimum, -3.0f)) return 3;
  float maximum = 9.0f;
  polygeist_cudnn_reduce_f32(3, 4, x, &maximum);
  if (!closef(maximum, 9.0f)) return 4;
  double xd[3] = {0.25, 0.5, 0.75};
  double sumd = 1.0;
  polygeist_cudnn_reduce_f64(0, 3, xd, &sumd);
  if (!closed(sumd, 2.5)) return 5;
  float matrix[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float trace = 2.0f;
  polygeist_cudnn_reduce_diagonal_f32(3, 4, 4, 1, matrix, &trace);
  if (!closef(trace, 20.0f)) return 6;
  puts("cudnn-reduction-reference: PASS");
  return 0;
}
