#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "atom.h"

void get_atom_extent(Vec3 *out_lo, Vec3 *out_hi, Atoms *atom);

enum { kWarmups = 5, kSamples = 5, kAtoms = 1048576 };

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

int main(void) {
  Atom *data = malloc((size_t)kAtoms * sizeof(Atom));
  if (!data)
    return 100;
  Vec3 expected_lo = {INFINITY, INFINITY, INFINITY};
  Vec3 expected_hi = {-INFINITY, -INFINITY, -INFINITY};
  for (int i = 0; i < kAtoms; ++i) {
    data[i].x = ((i * 17) % 100003 - 50001) * 0.001f;
    data[i].y = ((i * 29) % 70001 - 35000) * 0.002f;
    data[i].z = ((i * 43) % 50021 - 25010) * 0.004f;
    data[i].q = ((i % 31) - 15) * 0.01f;
    expected_lo.x = fminf(expected_lo.x, data[i].x);
    expected_lo.y = fminf(expected_lo.y, data[i].y);
    expected_lo.z = fminf(expected_lo.z, data[i].z);
    expected_hi.x = fmaxf(expected_hi.x, data[i].x);
    expected_hi.y = fmaxf(expected_hi.y, data[i].y);
    expected_hi.z = fmaxf(expected_hi.z, data[i].z);
  }
  Atoms atoms = {.atoms = data, .size = kAtoms};
  Vec3 lo, hi;
  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
    double begin = wall_ms();
    get_atom_extent(&lo, &hi, &atoms);
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }
  float max_abs = 0.0f;
#define CHECK_FIELD(value, expected) max_abs = fmaxf(max_abs, fabsf((value) - (expected)))
  CHECK_FIELD(lo.x, expected_lo.x);
  CHECK_FIELD(lo.y, expected_lo.y);
  CHECK_FIELD(lo.z, expected_lo.z);
  CHECK_FIELD(hi.x, expected_hi.x);
  CHECK_FIELD(hi.y, expected_hi.y);
  CHECK_FIELD(hi.z, expected_hi.z);
#undef CHECK_FIELD
  int ok = isfinite(max_abs) && max_abs == 0.0f;
  printf("BENCH_CORRECTNESS six_extrema_exact=%s max_abs=%g\n",
         ok ? "true" : "false", max_abs);
  printf("BENCH_RESULT workload=parboil_cutcp_atom_extent n=%d status=%s\n",
         kAtoms, ok ? "PASS" : "FAIL");
  free(data);
  return ok ? 0 : 1;
}
