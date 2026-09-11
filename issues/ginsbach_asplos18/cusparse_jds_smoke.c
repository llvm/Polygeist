#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  /* JDS row order is [original row 1, original row 0, original row 2]. */
  int row_counts[3] = {2, 2, 1};
  int diagonal_offsets[2] = {0, 3};
  int columns[5] = {0, 0, 2, 1, 2};
  float values[5] = {4.0f, 2.0f, 6.0f, 5.0f, 3.0f};
  int permutation[3] = {1, 0, 2};
  float x[3] = {1.0f, 2.0f, 3.0f};
  float y[3] = {-1.0f, -1.0f, -1.0f};
  const float expected[3] = {11.0f, 14.0f, 18.0f};

  polygeist_cublas_init();
  polygeist_cusparse_spmv_jds_f32_sized(
      3, 3, 3, row_counts, 2, diagonal_offsets, 5, columns, 5, values,
      3, permutation, 3, x, 3, y);
  polygeist_cublas_destroy();

  for (int i = 0; i < 3; ++i) {
    if (fabsf(y[i] - expected[i]) > 1.0e-5f) {
      fprintf(stderr, "FAIL y[%d]=%.9g expected %.9g\n", i, y[i], expected[i]);
      return 1;
    }
  }
  printf("PASS JDS-to-CSR cuSPARSE: [%.1f, %.1f, %.1f]\n", y[0], y[1], y[2]);
  return 0;
}
