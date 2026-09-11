// RUN: clang -O2 -DPOLYGEIST_CPU_USE_CBLAS -I%polygeist_src_root/runtime %s %polygeist_src_root/runtime/polygeist_cublas_rt_cpu.c -lopenblas -lm -o %t
// RUN: %t

#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

static int close_enough(double actual, double expected) {
  return fabs(actual - expected) <= 1.0e-12;
}

static int check_matrix(const char *name, const double *actual,
                        const double *expected) {
  for (int i = 0; i < 9; ++i) {
    if (!close_enough(actual[i], expected[i])) {
      fprintf(stderr, "%s[%d]: got %.17g, expected %.17g\n", name, i,
              actual[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  const double A[6] = {1, 2, 3, 4, 5, 6};
  const double B[6] = {2, 1, 0, 3, 4, 2};
  const double initial[9] = {1, 90, 91, 2, 3, 92, 4, 5, 6};
  double syrk[9];
  double syr2k[9];
  for (int i = 0; i < 9; ++i)
    syrk[i] = syr2k[i] = initial[i];

  polygeist_cublas_dsyrk_lower(3, 2, 2.0, A, 2, 0.5, syrk, 3);
  const double expected_syrk[9] = {
      10.5, 90, 91, 23, 51.5, 92, 36, 80.5, 125};
  if (check_matrix("dsyrk", syrk, expected_syrk))
    return 1;

  polygeist_cublas_dsyr2k_lower(3, 2, 2.0, A, 2, B, 2, 0.5, syr2k, 3);
  const double expected_syr2k[9] = {
      16.5, 90, 91, 33, 49.5, 92, 50, 78.5, 131};
  return check_matrix("dsyr2k", syr2k, expected_syr2k);
}
