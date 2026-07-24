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
