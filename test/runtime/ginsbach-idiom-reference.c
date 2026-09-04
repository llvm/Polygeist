// RUN: cc -std=gnu11 -I%polygeist_src_root/runtime %s %polygeist_src_root/runtime/polygeist_cublas_rt_cpu.c -lm -o %t && %t | FileCheck %s
// CHECK: ginsbach-idiom-reference: PASS

#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int closef(float a, float b) { return fabsf(a - b) < 1.0e-5f; }
static int closed(double a, double b) { return fabs(a - b) < 1.0e-12; }

int main(void) {
  double grid[64], rhs[64], residual[64] = {0};
  for (int i = 0; i < 64; ++i) grid[i] = 1.0, rhs[i] = 100.0;
  const double a[4] = {2.0, 0.0, 3.0, 5.0};
  polygeist_mg_resid_f64(grid, rhs, residual, 4, 4, 4, a);
  if (!closed(residual[21], 22.0)) return 1;

  double correction[64], unit_residual[64];
  for (int i = 0; i < 64; ++i)
    correction[i] = 2.0, unit_residual[i] = 1.0;
  const double c[4] = {1.0, 2.0, 3.0, 0.0};
  polygeist_mg_psinv_f64(unit_residual, correction, 4, 4, 4, c);
  if (!closed(correction[21], 51.0)) return 2;

  const int32_t values[7] = {1, 1, 1, 2, 2, -1, 9};
  uint8_t bins[4] = {0, 254, 3, 0};
  polygeist_histogram_saturating_u8(values, bins, 7, 4);
  if (bins[0] != 0 || bins[1] != 255 || bins[2] != 5 || bins[3] != 0)
    return 3;

  const float points[9] = {1, 0, 0, 0, 1, 0, -1, 0, 0};
  const float bounds[3] = {0.75f, 0.0f, -1.0f};
  int64_t pair_bins[4] = {0, 0, 0, 0};
  polygeist_tpacf_histogram_f32(points, 3, points, 3, 1,
                                pair_bins, 2, bounds);
  if (pair_bins[0] + pair_bins[1] + pair_bins[2] + pair_bins[3] != 3)
    return 4;

  const int32_t nzcnt[2] = {2, 1}, jptr[2] = {0, 2};
  const int32_t jcols[3] = {0, 1, 1}, perm[2] = {1, 0};
  const float jdata[3] = {2, 3, 4}, jx[2] = {5, 7};
  float jout[2] = {0, 0};
  polygeist_jds_spmv_f32(2, nzcnt, jptr, jcols, jdata, jx, perm, jout);
  if (!closef(jout[0], 21.0f) || !closef(jout[1], 38.0f)) return 5;

  const int32_t rowptr[3] = {0, 2, 3}, cols[3] = {0, 1, 1};
  const double data[3] = {2, 4, 3}, x[2] = {5, 7};
  double out[2] = {0, 0};
  polygeist_csr_spmv_f64(2, rowptr, cols, data, x, out);
  if (!closed(out[0], 38.0) || !closed(out[1], 21.0)) return 6;

  const float ga[6] = {1, 2, 3, 4, 5, 6};
  const float gb[6] = {7, 8, 9, 10, 11, 12};
  float gc[4] = {1, 1, 1, 1};
  polygeist_cublas_sgemm_transpose(2, 2, 3, 0, 0, 1, ga, 3, gb, 2,
                                    0, gc, 2);
  if (!closef(gc[0], 58) || !closef(gc[1], 64) ||
      !closef(gc[2], 139) || !closef(gc[3], 154)) return 7;

  puts("ginsbach-idiom-reference: PASS");
  return 0;
}
