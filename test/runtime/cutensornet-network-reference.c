#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void fill_metadata(int64_t metadata[31], int accumulate) {
  const int64_t values[31] = {
      1, 3, accumulate,
      2, 2, 2, 2,
      2, 3, 0, 3, 1, 2, // A[i,k]
      3, 2, 2, 2, 1, 1, // B[k,j]
      2, 2, 0, 2, 1, 1, // D[i,j]
      2, 2, 0, 2, 1, 1  // C[i,j]
  };
  for (int i = 0; i < 31; ++i) metadata[i] = values[i];
}

int main(void) {
  double a[6] = {1, 2, 3, 4, 5, 6};
  double b[6] = {7, 8, 9, 10, 11, 12};
  double d[4] = {0.5, 2.0, -1.0, 0.25};
  double c[4] = {100, 100, 100, 100};
  int64_t metadata[31];
  int64_t pointers[4] = {(int64_t)(uintptr_t)a, (int64_t)(uintptr_t)b,
                         (int64_t)(uintptr_t)d, (int64_t)(uintptr_t)c};
  const double expected[4] = {29, 128, -139, 38.5};

  fill_metadata(metadata, 0);
  polygeist_cutensornet_network_f64(pointers, metadata);
  for (int i = 0; i < 4; ++i)
    if (fabs(c[i] - expected[i]) > 1.0e-12) return 1;

  fill_metadata(metadata, 1);
  polygeist_cutensornet_network_f64(pointers, metadata);
  for (int i = 0; i < 4; ++i)
    if (fabs(c[i] - 2.0 * expected[i]) > 1.0e-12) return 2;

  puts("cutensornet-network-reference: PASS");
  return 0;
}
