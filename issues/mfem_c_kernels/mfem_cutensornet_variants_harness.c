#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
  MAX_RANK = 64,
  TENSOR_FIELDS = 3 * MAX_RANK,
  METADATA_SIZE = 3 + 3 * TENSOR_FIELDS,
  MAX_ELEMENTS = 4096
};

static void set_tensor(int64_t metadata[METADATA_SIZE], int tensor,
                       int64_t rank, const int64_t *extents,
                       const int64_t *strides, const int64_t *modes) {
  metadata[tensor] = rank;
  int64_t base = 3 + tensor * TENSOR_FIELDS;
  for (int64_t dim = 0; dim < MAX_RANK; ++dim) {
    metadata[base + dim] = dim < rank ? extents[dim] : 1;
    metadata[base + MAX_RANK + dim] = dim < rank ? strides[dim] : 0;
    metadata[base + 2 * MAX_RANK + dim] =
        dim < rank ? modes[dim] : -1;
  }
}

static int64_t tensor_span(const int64_t metadata[METADATA_SIZE], int tensor) {
  int64_t rank = metadata[tensor];
  int64_t base = 3 + tensor * TENSOR_FIELDS;
  int64_t span = 1;
  for (int64_t dim = 0; dim < rank; ++dim)
    span += (metadata[base + dim] - 1) *
            metadata[base + MAX_RANK + dim];
  return span;
}

static void reference_contraction(
    const double *a, const double *b, double *c,
    const int64_t metadata[METADATA_SIZE]) {
  int64_t mode_extents[MAX_RANK];
  int present[3][MAX_RANK] = {{0}};
  for (int mode = 0; mode < MAX_RANK; ++mode)
    mode_extents[mode] = 1;
  for (int tensor = 0; tensor < 3; ++tensor) {
    int64_t rank = metadata[tensor];
    int64_t base = 3 + tensor * TENSOR_FIELDS;
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t mode = metadata[base + 2 * MAX_RANK + dim];
      mode_extents[mode] = metadata[base + dim];
      present[tensor][mode] = 1;
    }
  }

  int64_t total = 1;
  for (int mode = 0; mode < MAX_RANK; ++mode)
    total *= mode_extents[mode];
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t coordinates[MAX_RANK];
    int64_t remaining = linear;
    for (int mode = MAX_RANK - 1; mode >= 0; --mode) {
      coordinates[mode] = remaining % mode_extents[mode];
      remaining /= mode_extents[mode];
    }

    int64_t offsets[3] = {0, 0, 0};
    for (int tensor = 0; tensor < 3; ++tensor) {
      int64_t rank = metadata[tensor];
      int64_t base = 3 + tensor * TENSOR_FIELDS;
      for (int64_t dim = 0; dim < rank; ++dim)
        offsets[tensor] +=
            coordinates[metadata[base + 2 * MAX_RANK + dim]] *
            metadata[base + MAX_RANK + dim];
    }

    int first_reduction_point = 1;
    for (int mode = 0; mode < MAX_RANK; ++mode)
      if (!present[2][mode] && (present[0][mode] || present[1][mode]) &&
          coordinates[mode] != 0)
        first_reduction_point = 0;
    if (first_reduction_point)
      c[offsets[2]] = 0.0;
    c[offsets[2]] += a[offsets[0]] * b[offsets[1]];
  }
}

static int run_case(const char *name,
                    const int64_t metadata[METADATA_SIZE]) {
  static double a[MAX_ELEMENTS], b[MAX_ELEMENTS];
  static double expected[MAX_ELEMENTS], actual[MAX_ELEMENTS];
  int64_t spans[3] = {
      tensor_span(metadata, 0),
      tensor_span(metadata, 1),
      tensor_span(metadata, 2),
  };
  if (spans[0] > MAX_ELEMENTS || spans[1] > MAX_ELEMENTS ||
      spans[2] > MAX_ELEMENTS)
    return 1;

  for (int64_t i = 0; i < spans[0]; ++i)
    a[i] = (double)((i * 7 + 3) % 29 - 14) / 11.0;
  for (int64_t i = 0; i < spans[1]; ++i)
    b[i] = (double)((i * 5 + 1) % 23 - 11) / 13.0;
  for (int64_t i = 0; i < spans[2]; ++i)
    expected[i] = actual[i] = -777.0;

  reference_contraction(a, b, expected, metadata);
  polygeist_cutensornet_contraction2_f64(a, b, actual, metadata);

  double max_error = 0.0;
  for (int64_t i = 0; i < spans[2]; ++i) {
    double error = fabs(actual[i] - expected[i]);
    if (error > max_error)
      max_error = error;
  }
  printf("%s span=(%ld,%ld,%ld) max_error=%.17g %s\n", name,
         (long)spans[0], (long)spans[1], (long)spans[2], max_error,
         max_error <= 1.0e-11 ? "PASS" : "FAIL");
  return max_error > 1.0e-11;
}

int main(void) {
  int failures = 0;

  {
    int64_t metadata[METADATA_SIZE] = {0};
    const int64_t ae[] = {2, 3, 4, 2};
    const int64_t as[] = {24, 8, 2, 1};
    const int64_t am[] = {0, 1, 4, 2};
    const int64_t be[] = {2, 2, 2, 3, 4};
    const int64_t bs[] = {48, 24, 12, 4, 1};
    const int64_t bm[] = {0, 3, 2, 1, 4};
    const int64_t ce[] = {2, 3, 2, 2};
    const int64_t cs[] = {12, 4, 2, 1};
    const int64_t cm[] = {0, 1, 3, 2};
    set_tensor(metadata, 0, 4, ae, as, am);
    set_tensor(metadata, 1, 5, be, bs, bm);
    set_tensor(metadata, 2, 4, ce, cs, cm);
    failures += run_case("r4r5r4", metadata);
  }

  {
    int64_t metadata[METADATA_SIZE] = {0};
    const int64_t ae[] = {2, 3, 2, 2, 4};
    const int64_t as[] = {96, 32, 16, 8, 1};
    const int64_t am[] = {0, 1, 2, 3, 4};
    const int64_t be[] = {2, 3, 4, 2};
    const int64_t bs[] = {24, 8, 2, 1};
    const int64_t bm[] = {0, 1, 4, 3};
    const int64_t ce[] = {2, 3, 2, 2};
    const int64_t cs[] = {12, 4, 2, 1};
    const int64_t cm[] = {0, 1, 2, 3};
    set_tensor(metadata, 0, 5, ae, as, am);
    set_tensor(metadata, 1, 4, be, bs, bm);
    set_tensor(metadata, 2, 4, ce, cs, cm);
    failures += run_case("r5r4r4", metadata);
  }

  {
    int64_t metadata[METADATA_SIZE] = {0};
    const int64_t ae[] = {2, 3, 2, 4};
    const int64_t as[] = {24, 8, 4, 1};
    const int64_t am[] = {0, 1, 3, 4};
    const int64_t be[] = {2, 2, 4};
    const int64_t bs[] = {8, 4, 1};
    const int64_t bm[] = {2, 3, 4};
    const int64_t ce[] = {2, 3, 2, 2};
    const int64_t cs[] = {12, 4, 2, 1};
    const int64_t cm[] = {0, 1, 2, 3};
    set_tensor(metadata, 0, 4, ae, as, am);
    set_tensor(metadata, 1, 3, be, bs, bm);
    set_tensor(metadata, 2, 4, ce, cs, cm);
    failures += run_case("r5r5r4_broadcast_compacted", metadata);
  }

  {
    int64_t metadata[METADATA_SIZE] = {0};
    const int64_t ae[] = {2, 3, 5, 4};
    const int64_t as[] = {60, 20, 4, 1};
    const int64_t am[] = {0, 1, 2, 3};
    const int64_t be[] = {2, 3, 5, 4};
    const int64_t bs[] = {60, 20, 4, 1};
    const int64_t bm[] = {0, 1, 2, 3};
    const int64_t ce[] = {2, 3, 5};
    const int64_t cs[] = {15, 5, 1};
    const int64_t cm[] = {0, 1, 2};
    set_tensor(metadata, 0, 4, ae, as, am);
    set_tensor(metadata, 1, 4, be, bs, bm);
    set_tensor(metadata, 2, 3, ce, cs, cm);
    failures += run_case("r4r4r3_2d", metadata);
  }

  {
    int64_t metadata[METADATA_SIZE] = {0};
    const int64_t ae[] = {2, 4, 3};
    const int64_t as[] = {12, 3, 1};
    const int64_t am[] = {0, 3, 1};
    const int64_t be[] = {2, 3, 5, 4};
    const int64_t bs[] = {60, 20, 4, 1};
    const int64_t bm[] = {0, 1, 2, 3};
    const int64_t ce[] = {2, 3, 5};
    const int64_t cs[] = {15, 5, 1};
    const int64_t cm[] = {0, 1, 2};
    set_tensor(metadata, 0, 3, ae, as, am);
    set_tensor(metadata, 1, 4, be, bs, bm);
    set_tensor(metadata, 2, 3, ce, cs, cm);
    failures += run_case("r3r4r3_2d_broadcast_compacted", metadata);
  }

  printf("mfem_cutensornet_variants failures=%d\n", failures);
  return failures != 0;
}
