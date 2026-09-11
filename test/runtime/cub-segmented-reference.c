#include "polygeist_cublas_rt.h"
#include <stdio.h>

int main(void) {
  const int x[12] = {1, 2, -3, 4, 1, 0, 3, 4, 5, 6, 7, 8};
  int out[3] = {0, 0, 0};
  polygeist_cub_segmented_reduce_i32(0, 3, 4, x, out);
  if (out[0] != 1 || out[1] != 0 || out[2] != 1) return 1;
  polygeist_cub_segmented_reduce_i32(1, 3, 4, x, out);
  if (out[0] != 1 || out[1] != 1 || out[2] != 1) return 2;
  polygeist_cub_segmented_reduce_i32(2, 3, 4, x, out);
  if (out[0] != (1 ^ 2 ^ -3 ^ 4) || out[1] != (1 ^ 0 ^ 3 ^ 4) ||
      out[2] != (5 ^ 6 ^ 7 ^ 8)) return 3;
  const int lengths[3] = {2, 3, 0};
  const float xf[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float outf[3] = {-1, -1, -1};
  polygeist_cub_segmented_prefix_sum_f32(3, 4, xf, lengths, outf);
  if (outf[0] != 3.0f || outf[1] != 18.0f || outf[2] != 0.0f) return 4;
  polygeist_cub_segmented_prefix_logical_and_i32(3, 4, x, lengths, out);
  if (out[0] != 1 || out[1] != 0 || out[2] != 1) return 5;
  puts("cub-segmented-reference: PASS");
  return 0;
}
