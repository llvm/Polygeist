#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *page_buffer(void) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, 65536, 65536) != 0) return NULL;
  memset(ptr, 0, 65536);
  return ptr;
}

static int closef(float a, float b) { return fabsf(a - b) < 1.0e-5f; }
static int closed(double a, double b) { return fabs(a - b) < 1.0e-12; }

int main(void) {
  double *zero_1d = page_buffer(), *zero_2d = page_buffer();
  if (!zero_1d || !zero_2d) return 99;
  for (int i = 0; i < 12; ++i) zero_1d[i] = 7.0, zero_2d[i] = 9.0;
  polygeist_cublas_memset_zero_1d(12, zero_1d);
  polygeist_cublas_memset_zero_2d(3, 4, zero_2d, 4);
  for (int i = 0; i < 12; ++i)
    if (zero_1d[i] != 0.0 || zero_2d[i] != 0.0) return 9;

  double *grid = page_buffer(), *rhs = page_buffer(), *residual = page_buffer();
  double *a = page_buffer(), *correction = page_buffer();
  double *unit_residual = page_buffer(), *coef = page_buffer();
  if (!grid || !rhs || !residual || !a || !correction || !unit_residual ||
      !coef) return 100;
  for (int i = 0; i < 64; ++i) grid[i] = 1.0, rhs[i] = 100.0;
  a[0] = 2.0; a[2] = 3.0; a[3] = 5.0;
  polygeist_mg_resid_f64(grid, rhs, residual, 4, 4, 4, a);
  if (!closed(residual[21], 22.0)) return 1;
  for (int i = 0; i < 64; ++i) correction[i] = 2.0, unit_residual[i] = 1.0;
  coef[0] = 1.0; coef[1] = 2.0; coef[2] = 3.0;
  polygeist_mg_psinv_f64(unit_residual, correction, 4, 4, 4, coef);
  if (!closed(correction[21], 51.0)) return 2;

  int32_t *values = page_buffer();
  uint8_t *bins = page_buffer();
  if (!values || !bins) return 101;
  const int32_t values_init[7] = {1, 1, 1, 2, 2, -1, 9};
  memcpy(values, values_init, sizeof(values_init));
  bins[1] = 254; bins[2] = 3;
  polygeist_histogram_saturating_u8(values, bins, 7, 4);
  if (bins[0] != 0 || bins[1] != 255 || bins[2] != 5 || bins[3] != 0)
    return 3;

  float *points = page_buffer(), *bounds = page_buffer();
  int64_t *pair_bins = page_buffer();
  if (!points || !bounds || !pair_bins) return 102;
  const float points_init[9] = {1, 0, 0, 0, 1, 0, -1, 0, 0};
  memcpy(points, points_init, sizeof(points_init));
  bounds[0] = 0.75f; bounds[1] = 0.0f; bounds[2] = -1.0f;
  polygeist_tpacf_histogram_f32(points, 3, points, 3, 1,
                                pair_bins, 2, bounds);
  if (pair_bins[0] + pair_bins[1] + pair_bins[2] + pair_bins[3] != 3)
    return 4;

  int32_t *nzcnt = page_buffer(), *jptr = page_buffer();
  int32_t *jcols = page_buffer(), *perm = page_buffer();
  float *jdata = page_buffer(), *jx = page_buffer(), *jout = page_buffer();
  if (!nzcnt || !jptr || !jcols || !perm || !jdata || !jx || !jout)
    return 103;
  nzcnt[0] = 2; nzcnt[1] = 1; jptr[0] = 0; jptr[1] = 2;
  jcols[0] = 0; jcols[1] = 1; jcols[2] = 1;
  perm[0] = 1; perm[1] = 0;
  jdata[0] = 2; jdata[1] = 3; jdata[2] = 4; jx[0] = 5; jx[1] = 7;
  polygeist_jds_spmv_f32_sized(2, nzcnt, 2, jptr, 3, jcols, 3, jdata,
                                2, jx, perm, 2, jout);
  if (!closef(jout[0], 21.0f) || !closef(jout[1], 38.0f)) return 5;

  int32_t *rowptr = page_buffer(), *cols = page_buffer();
  double *data = page_buffer(), *x = page_buffer(), *out = page_buffer();
  if (!rowptr || !cols || !data || !x || !out) return 104;
  rowptr[0] = 0; rowptr[1] = 2; rowptr[2] = 3;
  cols[0] = 0; cols[1] = 1; cols[2] = 1;
  data[0] = 2; data[1] = 4; data[2] = 3; x[0] = 5; x[1] = 7;
  polygeist_csr_spmv_f64_sized(2, 3, rowptr, 3, cols, 3, data,
                                2, x, 2, out);
  if (!closed(out[0], 38.0) || !closed(out[1], 21.0)) return 6;

  float *taps[7];
  for (int tap = 0; tap < 7; ++tap) {
    taps[tap] = page_buffer();
    if (!taps[tap]) return 105;
    taps[tap][0] = (float)(tap + 1);
    taps[tap][1] = (float)(tap + 2);
  }
  float *stencil_out = page_buffer();
  if (!stencil_out) return 106;
  polygeist_custom_stencil3d_7pt_flat_f32(
      2, taps[0], taps[1], taps[2], taps[3], taps[4], taps[5], taps[6],
      NULL, NULL, stencil_out, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1);
  if (!closef(stencil_out[0], 28.0f) || !closef(stencil_out[1], 35.0f))
    return 7;

  float *ga = page_buffer(), *gb = page_buffer(), *gc = page_buffer();
  if (!ga || !gb || !gc) return 107;
  const float ga_init[6] = {1, 2, 3, 4, 5, 6};
  const float gb_init[6] = {7, 8, 9, 10, 11, 12};
  memcpy(ga, ga_init, sizeof(ga_init)); memcpy(gb, gb_init, sizeof(gb_init));
  for (int i = 0; i < 4; ++i) gc[i] = 1.0f;
  polygeist_cublas_sgemm_transpose(2, 2, 3, 0, 0, 2, ga, 3, gb, 2,
                                    3, gc, 2);
  if (!closef(gc[0], 119) || !closef(gc[1], 131) ||
      !closef(gc[2], 281) || !closef(gc[3], 311)) return 8;

  puts("ginsbach-runtime-cuda-smoke: PASS");
  return 0;
}
