#include "polygeist_cuda_graph_rt.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern const unsigned char _binary_generated_scale_cubin_start[];
extern const unsigned char _binary_generated_scale_cubin_end[];

void *mgpuModuleLoad(void *data, size_t data_size);
void mgpuModuleUnload(void *module);
void *mgpuModuleGetFunction(void *module, const char *name);
void mgpuLaunchKernel(void *function, intptr_t grid_x, intptr_t grid_y,
                      intptr_t grid_z, intptr_t block_x, intptr_t block_y,
                      intptr_t block_z, int32_t shared_memory_bytes,
                      void *stream, void **params, void **extra,
                      size_t params_count);
void *mgpuStreamCreate(void);
void mgpuStreamSynchronize(void *stream);
void mgpuStreamDestroy(void *stream);

void polygeist_cublas_destroy(void);
void polygeist_cublas_memset_zero_1d(int32_t n, double *values);
void polygeist_cublas_daxpy_unit(int32_t n, const double *x, double *y);

static void check_cuda(cudaError_t status, const char *where) {
  if (status == cudaSuccess)
    return;
  fprintf(stderr, "%s failed: %s\n", where, cudaGetErrorString(status));
  exit(2);
}

static double now_ms(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return 1000.0 * (double)time.tv_sec + 1.0e-6 * (double)time.tv_nsec;
}

static void generated_scale_device(double *values, int64_t n, double scale) {
  size_t image_size = (size_t)(_binary_generated_scale_cubin_end -
                               _binary_generated_scale_cubin_start);
  void *module = mgpuModuleLoad((void *)_binary_generated_scale_cubin_start,
                                image_size);
  void *function = mgpuModuleGetFunction(
      module, "polygeist_generated_scale_kernel");
  void *stream = mgpuStreamCreate();
  void *params[] = {&values, &n, &scale};
  mgpuLaunchKernel(function, (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream,
                   params, NULL, 3);
  // Standard synchronous gpu.launch_func lowering emits these calls. The
  // Polygeist runtime suppresses this synchronization while a pipeline/graph
  // scope is active and keeps module ownership in its cache.
  mgpuStreamSynchronize(stream);
  mgpuStreamDestroy(stream);
  mgpuModuleUnload(module);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 4096;
  int iterations = argc > 2 ? atoi(argv[2]) : 5000;
  if (n <= 0 || iterations < 3)
    return 2;

  double *host_x = (double *)malloc((size_t)n * sizeof(double));
  double *host_y = (double *)malloc((size_t)n * sizeof(double));
  for (int i = 0; i < n; ++i)
    host_x[i] = 0.25 + (double)(i % 97) * 0.001;

  double *device_x = NULL;
  double *device_y = NULL;
  check_cuda(cudaMalloc((void **)&device_x, (size_t)n * sizeof(double)),
             "cudaMalloc x");
  check_cuda(cudaMalloc((void **)&device_y, (size_t)n * sizeof(double)),
             "cudaMalloc y");
  check_cuda(cudaMemcpy(device_x, host_x, (size_t)n * sizeof(double),
                        cudaMemcpyHostToDevice),
             "copy x");

  double start_ms = now_ms();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (polygeist_cuda_graph_begin(23)) {
      polygeist_cublas_memset_zero_1d(n, device_y);
      polygeist_cublas_daxpy_unit(n, device_x, device_y);
      generated_scale_device(device_y, n, 2.0);
      polygeist_cublas_daxpy_unit(n, device_x, device_y);
      polygeist_cuda_graph_end(23);
    }
  }
  double elapsed_ms = now_ms() - start_ms;

  check_cuda(cudaMemcpy(host_y, device_y, (size_t)n * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "copy y");
  double max_error = 0.0;
  for (int i = 0; i < n; ++i)
    max_error = fmax(max_error, fabs(host_y[i] - 3.0 * host_x[i]));

  printf("mixed_mgpu_graph correctness=%s max_error=%.3e n=%d "
         "iterations=%d total_ms=%.6f per_iteration_us=%.6f\n",
         max_error < 1.0e-12 ? "PASS" : "FAIL", max_error, n, iterations,
         elapsed_ms, 1000.0 * elapsed_ms / (double)iterations);
  cudaFree(device_x);
  cudaFree(device_y);
  free(host_x);
  free(host_y);
  polygeist_cublas_destroy();
  return max_error < 1.0e-12 ? 0 : 1;
}
