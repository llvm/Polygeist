// Fixed C ABI for CUB's seeded inclusive scan. Inputs are device pointers;
// the generated typed adapter owns any host/device marshalling.
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include "polygeist_cuda_graph_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist CUB %s: %s\n", operation, cudaGetErrorString(status));
    std::abort();
  }
}

template<class Op>
static void cub_scan_inclusive_seed(
    int32_t n, float seed, const float *input, float *output, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  int device;
  require_cuda(cudaGetDevice(&device), "device");
  struct Scratch { int device; void *pointer; size_t capacity; Scratch *next; };
  static thread_local Scratch *head = nullptr;
  Scratch *scratch = head;
  while (scratch && scratch->device != device) scratch = scratch->next;
  if (!scratch) { scratch = new Scratch{device, nullptr, 0, head}; head = scratch; }
  size_t bytes = 0;
  require_cuda(cub::DeviceScan::InclusiveScanInit(nullptr, bytes, input, output,
      op, seed, n, stream), "scratch query");
  if (bytes > scratch->capacity) {
    // This is reached only when reserving a larger shape, before replay/capture.
    require_cuda(cudaStreamSynchronize(stream), "scratch synchronization");
    require_cuda(cudaFree(scratch->pointer), "scratch free");
    scratch->pointer = nullptr; scratch->capacity = 0;
    require_cuda(cudaMalloc(&scratch->pointer, bytes), "scratch allocation");
    scratch->capacity = bytes;
  }
  require_cuda(cub::DeviceScan::InclusiveScanInit(scratch->pointer, bytes, input,
      output, op, seed, n, stream), "inclusive scan");
  const char *trace = std::getenv("POLYGEIST_CUB_TRACE");
  if (trace && *trace && std::strcmp(trace, "0"))
    std::fprintf(stderr, "POLYGEIST_CUB_CALL entry=DeviceScanInclusiveScanInit n=%d\n", n);
}

extern "C" void polygeist_cub_scan_inclusive_seed_f32(
    int32_t n, float seed, const float *input, float *output) {
  cub_scan_inclusive_seed(n, seed, input, output, cuda::std::plus<float>{});
}
extern "C" void polygeist_cub_scan_product_seed_f32(
    int32_t n, float seed, const float *input, float *output) {
  cub_scan_inclusive_seed(n, seed, input, output, cuda::std::multiplies<float>{});
}

// Fixed typed ABI for CUB DeviceReduce. The generated adapter marshals buffers.
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include "polygeist_cuda_graph_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void reduce_require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "polygeist CUB %s: %s\n", operation, cudaGetErrorString(status));
    std::abort();
  }
}

template<class T, class Op>
static void cub_reduce(int32_t n, T seed, const T *input, T *output, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  int device;
  reduce_require_cuda(cudaGetDevice(&device), "device");
  struct Scratch { int device; void *pointer; size_t capacity; Scratch *next; };
  static thread_local Scratch *head = nullptr;
  Scratch *scratch = head;
  while (scratch && scratch->device != device) scratch = scratch->next;
  if (!scratch) { scratch = new Scratch{device, nullptr, 0, head}; head = scratch; }
  size_t bytes = 0;
  reduce_require_cuda(cub::DeviceReduce::Reduce(nullptr, bytes, input, output,
      n, op, seed, stream), "reduce scratch query");
  if (bytes > scratch->capacity) {
    reduce_require_cuda(cudaStreamSynchronize(stream), "reduce scratch synchronization");
    reduce_require_cuda(cudaFree(scratch->pointer), "reduce scratch free");
    scratch->pointer = nullptr; scratch->capacity = 0;
    reduce_require_cuda(cudaMalloc(&scratch->pointer, bytes), "reduce scratch allocation");
    scratch->capacity = bytes;
  }
  reduce_require_cuda(cub::DeviceReduce::Reduce(scratch->pointer, bytes, input,
      output, n, op, seed, stream), "reduce");
  const char *trace = std::getenv("POLYGEIST_CUB_TRACE");
  if (trace && *trace && std::strcmp(trace, "0"))
    std::fprintf(stderr, "POLYGEIST_CUB_CALL entry=DeviceReduce n=%d bytes=%zu\n", n, sizeof(T));
}

extern "C" void polygeist_cub_reduce_sum_f32(int32_t n, float seed, const float *x, float *y) {
  cub_reduce(n, seed, x, y, cuda::std::plus<float>{});
}
extern "C" void polygeist_cub_reduce_sum_f64(int32_t n, double seed, const double *x, double *y) {
  cub_reduce(n, seed, x, y, cuda::std::plus<double>{});
}
extern "C" void polygeist_cub_reduce_product_f32(int32_t n, float seed, const float *x, float *y) {
  cub_reduce(n, seed, x, y, cuda::std::multiplies<float>{});
}

// CUB exposes device output storage; this C ABI returns the completed scalar.
// No scalar computation is reimplemented here.
template<class T, class Op>
static T cub_reduce_value(int32_t n, T seed, const T *input, Op op) {
  cudaStream_t stream = (cudaStream_t)polygeist_cuda_graph_stream();
  cudaStreamCaptureStatus capture;
  reduce_require_cuda(cudaStreamIsCapturing(stream, &capture), "scalar capture query");
  if (capture != cudaStreamCaptureStatusNone) {
    std::fprintf(stderr, "polygeist CUB scalar return requires a completed device result\n");
    std::abort();
  }
  int device;
  reduce_require_cuda(cudaGetDevice(&device), "scalar result device");
  struct Result { int device; T *pointer; Result *next; };
  static thread_local Result *head = nullptr;
  Result *result = head;
  while (result && result->device != device) result = result->next;
  if (!result) {
    result = new Result{device, nullptr, head};
    reduce_require_cuda(cudaMalloc(&result->pointer, sizeof(T)), "scalar result allocation");
    head = result;
  }
  cub_reduce(n, seed, input, result->pointer, op);
  T value;
  reduce_require_cuda(cudaMemcpyAsync(&value, result->pointer, sizeof(T),
                                     cudaMemcpyDeviceToHost, stream), "scalar result copy");
  reduce_require_cuda(cudaStreamSynchronize(stream), "scalar result completion");
  return value;
}
extern "C" float polygeist_cub_reduce_value_sum_f32(int32_t n, float seed, const float *input) {
  return cub_reduce_value(n, seed, input, cuda::std::plus<float>{});
}
extern "C" double polygeist_cub_reduce_value_sum_f64(int32_t n, double seed, const double *input) {
  return cub_reduce_value(n, seed, input, cuda::std::plus<double>{});
}
