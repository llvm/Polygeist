#include "../../runtime/polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  enum { BATCH = 2, M = 2, N = 2, K = 3 };
  const double a[BATCH * M * K] = {
      1, 2, 3, 4, 5, 6,
      -1, 2, 0, 3, -2, 1};
  const double b[BATCH * K * N] = {
      7, 8, 9, 10, 11, 12,
      2, -1, 4, 3, -2, 5};
  double c[BATCH * M * N] = {
      100, 101, 102, 103,
      20, 21, 22, 23};
  const double expected[BATCH * M * N] = {
      42, 37, -37, -51,
      14, 14, 26, 27};

  polygeist_cublas_dgemm_strided_batched_subtract(
      BATCH, M, N, K, a, b, c);
  for (int i = 0; i < BATCH * M * N; ++i) {
    if (fabs(c[i] - expected[i]) > 1.0e-12) {
      fprintf(stderr, "FAIL at %d: got %.17g expected %.17g\n",
              i, c[i], expected[i]);
      return 1;
    }
  }
  const double x[BATCH * K] = {1, 2, 3, -1, 4, 2};
  double y[BATCH * M] = {50, 60, 30, 40};
  const double expected_y[BATCH * M] = {36, 28, 21, 49};
  polygeist_cublas_dgemv_strided_batched_subtract(
      BATCH, M, K, a, x, y);
  for (int i = 0; i < BATCH * M; ++i) {
    if (fabs(y[i] - expected_y[i]) > 1.0e-12) {
      fprintf(stderr, "GEMV FAIL at %d: got %.17g expected %.17g\n",
              i, y[i], expected_y[i]);
      return 1;
    }
  }
  puts("PASS cublasDgemmStridedBatched subtract");
  puts("PASS cublasDgemvStridedBatched subtract");
  return 0;
}
