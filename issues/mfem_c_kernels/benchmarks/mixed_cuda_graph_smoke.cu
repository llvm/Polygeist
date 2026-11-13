#include "polygeist_cuda_graph_rt.h"

#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" {
void polygeist_cublas_destroy(void);
void polygeist_cublas_memset_zero_1d(int32_t n, double *values);
void polygeist_cublas_daxpy_unit(int32_t n, const double *x, double *y);
}

// This wrapper models the ABI emitted for an outlined compiler-generated GPU
// partition. It obtains (but does not own) the common Polygeist stream, so its
// launch can be ordered with library calls and captured in the same CUDA
// Graph.
__global__ static void generated_scale_kernel(double *values, int n,
                                              double scale) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n)
    values[i] *= scale;
}

extern "C" void generated_scale_device(double *values, int n,
                                        double scale) {
  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(polygeist_cuda_graph_stream());
  generated_scale_kernel<<<(n + 255) / 256, 256, 0, stream>>>(values, n,
                                                              scale);
}

static void check_cuda(cudaError_t status, const char *where) {
  if (status == cudaSuccess)
    return;
  std::fprintf(stderr, "%s failed: %s\n", where, cudaGetErrorString(status));
  std::exit(2);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  if (n <= 0 || iterations < 3)
    return 2;
  double *host_x = new double[n];
  double *host_y = new double[n];
  for (int i = 0; i < n; ++i)
    host_x[i] = 0.25 + (double)(i % 97) * 0.001;

  double *device_x = nullptr;
  double *device_y = nullptr;
  check_cuda(cudaMalloc(&device_x, (size_t)n * sizeof(double)), "cudaMalloc x");
  check_cuda(cudaMalloc(&device_y, (size_t)n * sizeof(double)), "cudaMalloc y");
  check_cuda(cudaMemcpy(device_x, host_x, (size_t)n * sizeof(double),
                        cudaMemcpyHostToDevice),
             "copy x");

  auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    // Warmup executes normally, the next invocation captures, and later
    // invocations replay this mixed sequence:
    //   CUDA memset -> cuBLAS DAXPY -> generated kernel -> cuBLAS DAXPY.
    if (polygeist_cuda_graph_begin(17)) {
      polygeist_cublas_memset_zero_1d(n, device_y);
      polygeist_cublas_daxpy_unit(n, device_x, device_y);
      generated_scale_device(device_y, n, 2.0);
      polygeist_cublas_daxpy_unit(n, device_x, device_y);
      polygeist_cuda_graph_end(17);
    }
  }
  auto stop = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();

  check_cuda(cudaMemcpy(host_y, device_y, (size_t)n * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "copy y");
  double max_error = 0.0;
  for (int i = 0; i < n; ++i)
    max_error = std::fmax(max_error, std::fabs(host_y[i] - 3.0 * host_x[i]));

  std::printf("mixed_cuda_graph correctness=%s max_error=%.3e n=%d "
              "iterations=%d total_ms=%.6f per_iteration_us=%.6f\n",
              max_error < 1.0e-12 ? "PASS" : "FAIL", max_error, n,
              iterations, elapsed_ms, 1000.0 * elapsed_ms / iterations);
  cudaFree(device_x);
  cudaFree(device_y);
  delete[] host_x;
  delete[] host_y;
  polygeist_cublas_destroy();
  return max_error < 1.0e-12 ? 0 : 1;
}
