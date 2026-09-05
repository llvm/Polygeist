// RUN: clang -O2 -I%polygeist_src_root/runtime %s %polygeist_src_root/runtime/polygeist_cublas_rt_cpu.c -lm -o %t && %t | FileCheck %s

#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int close_enough(double a, double b) {
  return fabs(a - b) < 1.0e-10;
}

int main(void) {
  const int32_t samples[] = {0, 1, 4, 7, 8, 15};
  int32_t histogram[4] = {-1, -1, -1, -1};
  polygeist_cub_histogram_even_i32_shift_zero(
      6, 4, samples, histogram, 2);
  const int32_t expected_histogram[] = {2, 2, 1, 1};
  for (int i = 0; i < 4; ++i)
    if (histogram[i] != expected_histogram[i]) return 1;

  const double lower[] = {
      2.0, 0.0, 0.0,
      1.0, 3.0, 0.0,
      4.0, -2.0, 1.0,
  };
  const double rhs[] = {2.0, 7.0, 3.0};
  double solution[3] = {0.0, 0.0, 0.0};
  polygeist_cublas_dtrsv_lower_row_major(3, lower, rhs, solution);
  if (!close_enough(solution[0], 1.0) ||
      !close_enough(solution[1], 2.0) ||
      !close_enough(solution[2], 3.0))
    return 2;

  double positive_definite[] = {
      4.0, 2.0, 2.0,
      2.0, 5.0, 1.0,
      2.0, 1.0, 3.0,
  };
  polygeist_cusolver_dpotrf_lower_row_major(3, positive_definite);
  const double expected_lower[] = {
      2.0, 0.0, 0.0,
      1.0, 2.0, 0.0,
      1.0, 0.0, sqrt(2.0),
  };
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j)
      if (!close_enough(positive_definite[i * 3 + j],
                        expected_lower[i * 3 + j]))
        return 3;
  puts("PASS");
  return 0;
}

// CHECK: PASS
