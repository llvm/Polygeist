#include "polygeist_stencil3d_7pt.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define CUDA_OK(call) do {                                                   \
  cudaError_t error = (call);                                                \
  if (error != cudaSuccess) {                                                \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                 cudaGetErrorString(error));                                 \
    return 100;                                                              \
  }                                                                          \
} while (0)

template <typename T>
static T *device_copy(const T *host, size_t count) {
  T *device = nullptr;
  if (cudaMalloc(&device, count * sizeof(T)) != cudaSuccess)
    return nullptr;
  if (host && cudaMemcpy(device, host, count * sizeof(T),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(device);
    return nullptr;
  }
  return device;
}

static bool closef(float a, float b) { return std::fabs(a - b) < 1.0e-5f; }
static bool closed(double a, double b) { return std::fabs(a - b) < 1.0e-12; }

int main() {
  double grid[64], rhs[64], residual[64] = {};
  for (int i = 0; i < 64; ++i) grid[i] = 1.0, rhs[i] = 100.0;
  const double a[4] = {2.0, 0.0, 3.0, 5.0};
  double *d_grid = device_copy(grid, 64), *d_rhs = device_copy(rhs, 64);
  double *d_residual = device_copy(residual, 64), *d_a = device_copy(a, 4);
  if (!d_grid || !d_rhs || !d_residual || !d_a) return 101;
  polygeist_mg_resid_f64_device(d_grid, d_rhs, d_residual, 4, 4, 4, d_a,
                                nullptr);
  CUDA_OK(cudaMemcpy(residual, d_residual, sizeof(residual),
                     cudaMemcpyDeviceToHost));
  if (!closed(residual[21], 22.0)) return 1;

  double correction[64], unit_residual[64];
  for (int i = 0; i < 64; ++i)
    correction[i] = 2.0, unit_residual[i] = 1.0;
  const double c[4] = {1.0, 2.0, 3.0, 0.0};
  double *d_correction = device_copy(correction, 64);
  double *d_unit_residual = device_copy(unit_residual, 64);
  double *d_c = device_copy(c, 4);
  if (!d_correction || !d_unit_residual || !d_c) return 102;
  polygeist_mg_psinv_f64_device(d_unit_residual, d_correction, 4, 4, 4, d_c,
                                nullptr);
  CUDA_OK(cudaMemcpy(correction, d_correction, sizeof(correction),
                     cudaMemcpyDeviceToHost));
  if (!closed(correction[21], 51.0)) return 2;

  const int32_t values[7] = {1, 1, 1, 2, 2, -1, 9};
  uint8_t bins[4] = {0, 254, 3, 0};
  int32_t *d_values = device_copy(values, 7);
  uint8_t *d_bins = device_copy(bins, 4);
  if (!d_values || !d_bins) return 103;
  polygeist_histogram_saturating_u8_device(d_values, d_bins, 7, 4, nullptr);
  CUDA_OK(cudaMemcpy(bins, d_bins, sizeof(bins), cudaMemcpyDeviceToHost));
  if (bins[0] != 0 || bins[1] != 255 || bins[2] != 5 || bins[3] != 0)
    return 3;

  const float points[9] = {1, 0, 0, 0, 1, 0, -1, 0, 0};
  const float bounds[3] = {0.75f, 0.0f, -1.0f};
  int64_t pair_bins[4] = {};
  float *d_points = device_copy(points, 9), *d_bounds = device_copy(bounds, 3);
  int64_t *d_pair_bins = device_copy(pair_bins, 4);
  if (!d_points || !d_bounds || !d_pair_bins) return 104;
  polygeist_tpacf_histogram_f32_device(d_points, 3, d_points, 3, 1,
                                       d_pair_bins, 2, d_bounds, nullptr);
  CUDA_OK(cudaMemcpy(pair_bins, d_pair_bins, sizeof(pair_bins),
                     cudaMemcpyDeviceToHost));
  if (pair_bins[0] + pair_bins[1] + pair_bins[2] + pair_bins[3] != 3)
    return 4;

  const int32_t nzcnt[2] = {2, 1}, jptr[2] = {0, 2};
  const int32_t jcols[3] = {0, 1, 1}, perm[2] = {1, 0};
  const float jdata[3] = {2, 3, 4}, jx[2] = {5, 7};
  float jout[2] = {};
  int32_t *d_nzcnt = device_copy(nzcnt, 2), *d_jptr = device_copy(jptr, 2);
  int32_t *d_jcols = device_copy(jcols, 3), *d_perm = device_copy(perm, 2);
  float *d_jdata = device_copy(jdata, 3), *d_jx = device_copy(jx, 2);
  float *d_jout = device_copy(jout, 2);
  if (!d_nzcnt || !d_jptr || !d_jcols || !d_perm || !d_jdata || !d_jx ||
      !d_jout) return 105;
  polygeist_jds_spmv_f32_device(2, d_nzcnt, d_jptr, d_jcols, d_jdata, d_jx,
                                d_perm, d_jout, nullptr);
  CUDA_OK(cudaMemcpy(jout, d_jout, sizeof(jout), cudaMemcpyDeviceToHost));
  if (!closef(jout[0], 21.0f) || !closef(jout[1], 38.0f)) return 5;

  const int32_t rowptr[3] = {0, 2, 3}, cols[3] = {0, 1, 1};
  const double data[3] = {2, 4, 3}, x[2] = {5, 7};
  double out[2] = {};
  int32_t *d_rowptr = device_copy(rowptr, 3), *d_cols = device_copy(cols, 3);
  double *d_data = device_copy(data, 3), *d_x = device_copy(x, 2);
  double *d_out = device_copy(out, 2);
  if (!d_rowptr || !d_cols || !d_data || !d_x || !d_out) return 106;
  polygeist_csr_spmv_f64_device(2, d_rowptr, d_cols, d_data, d_x, d_out,
                                nullptr);
  CUDA_OK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
  if (!closed(out[0], 38.0) || !closed(out[1], 21.0)) return 6;

  const float tap0[2] = {1, 2}, tap1[2] = {2, 3}, tap2[2] = {3, 4};
  const float tap3[2] = {4, 5}, tap4[2] = {5, 6}, tap5[2] = {6, 7};
  const float tap6[2] = {7, 8};
  float stencil_out[2] = {};
  float *d_tap0 = device_copy(tap0, 2), *d_tap1 = device_copy(tap1, 2);
  float *d_tap2 = device_copy(tap2, 2), *d_tap3 = device_copy(tap3, 2);
  float *d_tap4 = device_copy(tap4, 2), *d_tap5 = device_copy(tap5, 2);
  float *d_tap6 = device_copy(tap6, 2), *d_stencil_out = device_copy(stencil_out, 2);
  if (!d_tap0 || !d_tap1 || !d_tap2 || !d_tap3 || !d_tap4 || !d_tap5 ||
      !d_tap6 || !d_stencil_out) return 107;
  polygeist_custom_stencil3d_7pt_flat_f32_device(
      2, d_tap0, d_tap1, d_tap2, d_tap3, d_tap4, d_tap5, d_tap6,
      nullptr, nullptr, d_stencil_out, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
      nullptr);
  CUDA_OK(cudaMemcpy(stencil_out, d_stencil_out, sizeof(stencil_out),
                     cudaMemcpyDeviceToHost));
  if (!closef(stencil_out[0], 28.0f) || !closef(stencil_out[1], 35.0f))
    return 7;

  CUDA_OK(cudaDeviceSynchronize());
  std::puts("ginsbach-cuda-smoke: PASS");
  return 0;
}
