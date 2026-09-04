#include "polygeist_stencil3d_7pt.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define POLYGEIST_CUSTOM_CUDA_CHECK(call) do {                               \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "%s:%d cuda error: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      abort();                                                               \
    }                                                                        \
  } while (0)

__device__ __forceinline__ int64_t mg_offset(int32_t x, int32_t y, int32_t z,
                                              int32_t n1, int32_t n2) {
  return ((int64_t)z * n2 + y) * n1 + x;
}

__global__ void polygeist_mg_resid_kernel(
    const double *u, const double *v, double *r, int32_t n1, int32_t n2,
    int32_t n3, const double *a) {
  int32_t x = 1 + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t y = 1 + (int32_t)(blockIdx.y * blockDim.y + threadIdx.y);
  int32_t z = 1 + (int32_t)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= n1 - 1 || y >= n2 - 1 || z >= n3 - 1) return;
  int64_t p = mg_offset(x, y, z, n1, n2);
  int64_t sy = n1, sz = (int64_t)n1 * n2;
  double u1m = u[p-sy-1] + u[p+sy-1] + u[p-sz-1] + u[p+sz-1];
  double u1p = u[p-sy+1] + u[p+sy+1] + u[p-sz+1] + u[p+sz+1];
  double u2 = u[p-sz-sy] + u[p-sz+sy] + u[p+sz-sy] + u[p+sz+sy];
  double u2m = u[p-sz-sy-1] + u[p-sz+sy-1] + u[p+sz-sy-1] + u[p+sz+sy-1];
  double u2p = u[p-sz-sy+1] + u[p-sz+sy+1] + u[p+sz-sy+1] + u[p+sz+sy+1];
  r[p] = v[p] - a[0] * u[p] - a[2] * (u2 + u1m + u1p)
         - a[3] * (u2m + u2p);
}

__global__ void polygeist_mg_psinv_kernel(
    const double *r, double *u, int32_t n1, int32_t n2, int32_t n3,
    const double *c) {
  int32_t x = 1 + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t y = 1 + (int32_t)(blockIdx.y * blockDim.y + threadIdx.y);
  int32_t z = 1 + (int32_t)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= n1 - 1 || y >= n2 - 1 || z >= n3 - 1) return;
  int64_t p = mg_offset(x, y, z, n1, n2);
  int64_t sy = n1, sz = (int64_t)n1 * n2;
  double r1 = r[p-sy] + r[p+sy] + r[p-sz] + r[p+sz];
  double r1m = r[p-sy-1] + r[p+sy-1] + r[p-sz-1] + r[p+sz-1];
  double r1p = r[p-sy+1] + r[p+sy+1] + r[p-sz+1] + r[p+sz+1];
  double r2 = r[p-sz-sy] + r[p-sz+sy] + r[p+sz-sy] + r[p+sz+sy];
  u[p] += c[0] * r[p] + c[1] * (r[p-1] + r[p+1] + r1)
          + c[2] * (r2 + r1m + r1p);
}

extern "C" void polygeist_mg_resid_f64_device(
    const double *u, const double *v, double *r, int32_t n1, int32_t n2,
    int32_t n3, const double *a, void *cuda_stream) {
  if (n1 <= 2 || n2 <= 2 || n3 <= 2) return;
  dim3 block(8, 4, 4);
  dim3 grid((n1 - 2 + block.x - 1) / block.x,
            (n2 - 2 + block.y - 1) / block.y,
            (n3 - 2 + block.z - 1) / block.z);
  polygeist_mg_resid_kernel<<<grid, block, 0, (cudaStream_t)cuda_stream>>>(
      u, v, r, n1, n2, n3, a);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

extern "C" void polygeist_mg_psinv_f64_device(
    const double *r, double *u, int32_t n1, int32_t n2, int32_t n3,
    const double *c, void *cuda_stream) {
  if (n1 <= 2 || n2 <= 2 || n3 <= 2) return;
  dim3 block(8, 4, 4);
  dim3 grid((n1 - 2 + block.x - 1) / block.x,
            (n2 - 2 + block.y - 1) / block.y,
            (n3 - 2 + block.z - 1) / block.z);
  polygeist_mg_psinv_kernel<<<grid, block, 0, (cudaStream_t)cuda_stream>>>(
      r, u, n1, n2, n3, c);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

__global__ void polygeist_histogram_u32_kernel(const int32_t *values,
                                                int32_t count,
                                                int32_t num_bins,
                                                uint32_t *counts) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    int32_t bin = values[i];
    if ((uint32_t)bin < (uint32_t)num_bins)
      atomicAdd(counts + bin, 1u);
  }
}

__global__ void polygeist_histogram_seed_u32_kernel(const uint8_t *bins,
                                                     uint32_t *counts,
                                                     int32_t num_bins) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_bins) counts[i] = bins[i];
}

__global__ void polygeist_histogram_clamp_u8_kernel(const uint32_t *counts,
                                                     uint8_t *bins,
                                                     int32_t num_bins) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_bins) bins[i] = (uint8_t)(counts[i] > 255u ? 255u : counts[i]);
}

extern "C" void polygeist_histogram_saturating_u8_device(
    const int32_t *values, uint8_t *bins, int32_t count, int32_t num_bins,
    void *cuda_stream) {
  if (count <= 0 || num_bins <= 0) return;
  cudaStream_t stream = (cudaStream_t)cuda_stream;
  uint32_t *counts = nullptr;
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaMalloc((void **)&counts,
      (size_t)num_bins * sizeof(uint32_t)));
  int threads = 256;
  polygeist_histogram_seed_u32_kernel<<<
      (num_bins + threads - 1) / threads, threads, 0, stream>>>(
          bins, counts, num_bins);
  polygeist_histogram_u32_kernel<<<(count + threads - 1) / threads,
      threads, 0, stream>>>(values, count, num_bins, counts);
  polygeist_histogram_clamp_u8_kernel<<<(num_bins + threads - 1) / threads,
      threads, 0, stream>>>(counts, bins, num_bins);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaFree(counts));
}

__global__ void polygeist_tpacf_histogram_kernel(
    const float *data1, int32_t n1, const float *data2, int32_t n2,
    int32_t self, unsigned long long *bins, int32_t nbins,
    const float *bounds) {
  int32_t i = (int32_t)(blockIdx.y * blockDim.y + threadIdx.y);
  int32_t j = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n1 || j >= n2 || (self && (i >= n1 - 1 || j <= i))) return;
  float dot = data1[3*i] * data2[3*j] + data1[3*i+1] * data2[3*j+1]
              + data1[3*i+2] * data2[3*j+2];
  int32_t lo = 0, hi = nbins;
  while (hi > lo + 1) {
    int32_t k = (lo + hi) / 2;
    if (dot >= bounds[k]) hi = k; else lo = k;
  }
  int32_t bin = dot >= bounds[lo] ? lo : (dot < bounds[hi] ? hi + 1 : hi);
  atomicAdd(bins + bin, 1ULL);
}

extern "C" void polygeist_tpacf_histogram_f32_device(
    const float *data1, int32_t n1, const float *data2, int32_t n2,
    int32_t self, int64_t *bins, int32_t nbins, const float *bounds,
    void *cuda_stream) {
  if (self) { data2 = data1; n2 = n1; }
  if (n1 <= 0 || n2 <= 0) return;
  dim3 block(16, 16);
  dim3 grid((n2 + 15) / 16, (n1 + 15) / 16);
  polygeist_tpacf_histogram_kernel<<<grid, block, 0,
      (cudaStream_t)cuda_stream>>>(data1, n1, data2, n2, self,
      (unsigned long long *)bins, nbins, bounds);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

__global__ void polygeist_jds_spmv_kernel(
    int32_t rows, const int32_t *nzcnt, const int32_t *ptr,
    const int32_t *indices, const float *data, const float *x,
    const int32_t *perm, float *out) {
  int32_t row = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  float sum = 0.0f;
  for (int32_t k = 0; k < nzcnt[row]; ++k) {
    int32_t j = ptr[k] + row;
    sum += data[j] * x[indices[j]];
  }
  out[perm[row]] = sum;
}

extern "C" void polygeist_jds_spmv_f32_device(
    int32_t rows, const int32_t *nzcnt, const int32_t *ptr,
    const int32_t *indices, const float *data, const float *x,
    const int32_t *perm, float *out, void *cuda_stream) {
  int threads = 256;
  polygeist_jds_spmv_kernel<<<(rows + threads - 1) / threads, threads, 0,
      (cudaStream_t)cuda_stream>>>(rows, nzcnt, ptr, indices, data, x, perm, out);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

__global__ void polygeist_csr_spmv_kernel(
    int32_t rows, const int32_t *rowptr, const int32_t *cols,
    const double *data, const double *x, double *out) {
  int32_t row = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  double sum = 0.0;
  for (int32_t j = rowptr[row]; j < rowptr[row + 1]; ++j)
    sum += data[j] * x[cols[j]];
  out[row] = sum;
}

extern "C" void polygeist_csr_spmv_f64_device(
    int32_t rows, const int32_t *rowptr, const int32_t *cols,
    const double *data, const double *x, double *out, void *cuda_stream) {
  int threads = 256;
  polygeist_csr_spmv_kernel<<<(rows + threads - 1) / threads, threads, 0,
      (cudaStream_t)cuda_stream>>>(rows, rowptr, cols, data, x, out);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void polygeist_stencil3d_7pt_kernel(
    int32_t nx, int32_t ny, int32_t nz,
    const T *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const T *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const T *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    T *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    T base_center, T base_extra, T coeff_extra,
    T coeff_center,
    T coeff_xm, T coeff_xp,
    T coeff_ym, T coeff_yp,
    T coeff_zm, T coeff_zp) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t j = (int32_t)(blockIdx.y * blockDim.y + threadIdx.y);
  int32_t k = (int32_t)(blockIdx.z * blockDim.z + threadIdx.z);
  if (i >= nx || j >= ny || k >= nz)
    return;

  int64_t input_offset =
      (int64_t)i * input_stride_i +
      (int64_t)j * input_stride_j +
      (int64_t)k * input_stride_k;
  int64_t output_offset =
      (int64_t)i * output_stride_i +
      (int64_t)j * output_stride_j +
      (int64_t)k * output_stride_k;

  T center = input_center[input_offset];
  T extra_value = T(0);
  if (extra) {
    int64_t extra_offset =
        (int64_t)i * extra_stride_i +
        (int64_t)j * extra_stride_j +
        (int64_t)k * extra_stride_k;
    extra_value = extra[extra_offset];
  }

  T base = base_center * center;
  if (extra)
    base += base_extra * extra_value;

  T inner =
      coeff_center * center +
      coeff_xm * input_center[input_offset - input_stride_i] +
      coeff_xp * input_center[input_offset + input_stride_i] +
      coeff_ym * input_center[input_offset - input_stride_j] +
      coeff_yp * input_center[input_offset + input_stride_j] +
      coeff_zm * input_center[input_offset - input_stride_k] +
      coeff_zp * input_center[input_offset + input_stride_k];
  if (extra)
    inner += coeff_extra * extra_value;

  T scale = T(1);
  if (coeff) {
    int64_t coeff_offset =
        (int64_t)i * coeff_stride_i +
        (int64_t)j * coeff_stride_j +
        (int64_t)k * coeff_stride_k;
    scale = coeff[coeff_offset];
  }

  output[output_offset] = base + scale * inner;
}

template <typename T>
static void launch_stencil3d_7pt(
    int32_t nx, int32_t ny, int32_t nz,
    const T *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const T *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const T *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    T *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    T base_center, T base_extra, T coeff_extra,
    T coeff_center,
    T coeff_xm, T coeff_xp,
    T coeff_ym, T coeff_yp,
    T coeff_zm, T coeff_zp,
    void *cuda_stream) {
  if (nx <= 0 || ny <= 0 || nz <= 0)
    return;
  if (!input_center || !output) {
    fprintf(stderr, "polygeist custom stencil3d_7pt: null input/output\n");
    abort();
  }

  cudaStream_t stream = (cudaStream_t)cuda_stream;
  dim3 block(8, 8, 4);
  dim3 grid((uint32_t)((nx + block.x - 1) / block.x),
            (uint32_t)((ny + block.y - 1) / block.y),
            (uint32_t)((nz + block.z - 1) / block.z));
  polygeist_stencil3d_7pt_kernel<T><<<grid, block, 0, stream>>>(
      nx, ny, nz,
      input_center,
      input_stride_i, input_stride_j, input_stride_k,
      extra,
      extra_stride_i, extra_stride_j, extra_stride_k,
      coeff,
      coeff_stride_i, coeff_stride_j, coeff_stride_k,
      output,
      output_stride_i, output_stride_j, output_stride_k,
      base_center, base_extra, coeff_extra,
      coeff_center,
      coeff_xm, coeff_xp,
      coeff_ym, coeff_yp,
      coeff_zm, coeff_zp);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void polygeist_stencil3d_7pt_flat_kernel(
    int32_t N,
    const T *a0, const T *a1, const T *a2,
    const T *a3, const T *a4, const T *a5,
    const T *a6, const T *extra, const T *coeff, T *out,
    T base0, T base_extra, T coeff_extra,
    T c0, T c1, T c2, T c3, T c4, T c5, T c6) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N)
    return;
  T extra_value = extra ? extra[i] : T(0);
  T scale = coeff ? coeff[i] : T(1);
  T base = base0 * a0[i];
  if (extra)
    base += base_extra * extra_value;
  T inner = c0 * a0[i] + c1 * a1[i] + c2 * a2[i] +
            c3 * a3[i] + c4 * a4[i] + c5 * a5[i] + c6 * a6[i];
  if (extra)
    inner += coeff_extra * extra_value;
  out[i] = base + scale * inner;
}

template <typename T>
static void launch_stencil3d_7pt_flat(
    int32_t N,
    const T *a0, const T *a1, const T *a2,
    const T *a3, const T *a4, const T *a5,
    const T *a6, const T *extra, const T *coeff, T *out,
    T base0, T base_extra, T coeff_extra,
    T c0, T c1, T c2, T c3, T c4, T c5, T c6,
    void *cuda_stream) {
  if (N <= 0)
    return;
  if (!a0 || !a1 || !a2 || !a3 || !a4 || !a5 || !a6 || !out) {
    fprintf(stderr, "polygeist custom stencil3d_7pt_flat: null required input\n");
    abort();
  }
  cudaStream_t stream = (cudaStream_t)cuda_stream;
  int block = 256;
  int grid = (N + block - 1) / block;
  polygeist_stencil3d_7pt_flat_kernel<T><<<grid, block, 0, stream>>>(
      N, a0, a1, a2, a3, a4, a5, a6, extra, coeff, out,
      base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6);
  POLYGEIST_CUSTOM_CUDA_CHECK(cudaGetLastError());
}

extern "C" void polygeist_custom_stencil3d_7pt_f64(
    int32_t nx, int32_t ny, int32_t nz,
    const double *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const double *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const double *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    double *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    double base_center, double base_extra, double coeff_extra,
    double coeff_center,
    double coeff_xm, double coeff_xp,
    double coeff_ym, double coeff_yp,
    double coeff_zm, double coeff_zp,
    void *cuda_stream) {
  launch_stencil3d_7pt<double>(
      nx, ny, nz,
      input_center,
      input_stride_i, input_stride_j, input_stride_k,
      extra,
      extra_stride_i, extra_stride_j, extra_stride_k,
      coeff,
      coeff_stride_i, coeff_stride_j, coeff_stride_k,
      output,
      output_stride_i, output_stride_j, output_stride_k,
      base_center, base_extra, coeff_extra,
      coeff_center,
      coeff_xm, coeff_xp,
      coeff_ym, coeff_yp,
      coeff_zm, coeff_zp,
      cuda_stream);
}

extern "C" void polygeist_custom_stencil3d_7pt_flat_f64_device(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6,
    void *cuda_stream) {
  launch_stencil3d_7pt_flat<double>(
      N, a0, a1, a2, a3, a4, a5, a6, extra, coeff, out,
      base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6,
      cuda_stream);
}

extern "C" void polygeist_custom_stencil3d_7pt_f32(
    int32_t nx, int32_t ny, int32_t nz,
    const float *input_center,
    int64_t input_stride_i, int64_t input_stride_j, int64_t input_stride_k,
    const float *extra,
    int64_t extra_stride_i, int64_t extra_stride_j, int64_t extra_stride_k,
    const float *coeff,
    int64_t coeff_stride_i, int64_t coeff_stride_j, int64_t coeff_stride_k,
    float *output,
    int64_t output_stride_i, int64_t output_stride_j, int64_t output_stride_k,
    float base_center, float base_extra, float coeff_extra,
    float coeff_center,
    float coeff_xm, float coeff_xp,
    float coeff_ym, float coeff_yp,
    float coeff_zm, float coeff_zp,
    void *cuda_stream) {
  launch_stencil3d_7pt<float>(
      nx, ny, nz,
      input_center,
      input_stride_i, input_stride_j, input_stride_k,
      extra,
      extra_stride_i, extra_stride_j, extra_stride_k,
      coeff,
      coeff_stride_i, coeff_stride_j, coeff_stride_k,
      output,
      output_stride_i, output_stride_j, output_stride_k,
      base_center, base_extra, coeff_extra,
      coeff_center,
      coeff_xm, coeff_xp,
      coeff_ym, coeff_yp,
      coeff_zm, coeff_zp,
      cuda_stream);
}

extern "C" void polygeist_custom_stencil3d_7pt_flat_f32_device(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6,
    void *cuda_stream) {
  launch_stencil3d_7pt_flat<float>(
      N, a0, a1, a2, a3, a4, a5, a6, extra, coeff, out,
      base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6,
      cuda_stream);
}
