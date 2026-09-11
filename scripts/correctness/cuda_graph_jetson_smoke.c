// End-to-end smoke test for Polygeist's cached CUDA Graph scope.
// Compile this together with runtime/polygeist_cublas_rt_cuda.c, then run with
// POLYGEIST_CUDA_GRAPH=0 and =1. Device allocations deliberately stay stable.

#include "polygeist_cublas_rt.h"

#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return 1000.0 * (double)time.tv_sec + 1.0e-6 * (double)time.tv_nsec;
}

static void check_cuda(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return;
  fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
  exit(2);
}

int main(int argc, char **argv) {
  int32_t elements = argc > 1 ? (int32_t)strtol(argv[1], NULL, 10) : 4096;
  int iterations = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 1000;
  size_t bytes = (size_t)elements * sizeof(float);
  float *host = (float *)malloc(bytes);
  float *x = NULL;
  float *y = NULL;
  if (!host || elements <= 0 || iterations < 3)
    return 2;
  for (int32_t i = 0; i < elements; ++i)
    host[i] = 1.0f;
  check_cuda(cudaMalloc((void **)&x, bytes), "cudaMalloc(x)");
  check_cuda(cudaMalloc((void **)&y, bytes), "cudaMalloc(y)");
  check_cuda(cudaMemcpy(x, host, bytes, cudaMemcpyHostToDevice), "copy x");
  check_cuda(cudaMemset(y, 0, bytes), "clear y");

  double start = now_ms();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (polygeist_cuda_graph_begin(7)) {
      polygeist_cublas_saxpby(elements, 1.0f, x, 1.0f, y);
      polygeist_cuda_graph_end(7);
    }
  }
  double elapsed = now_ms() - start;

  check_cuda(cudaMemcpy(host, y, bytes, cudaMemcpyDeviceToHost), "copy y");
  float max_error = 0.0f;
  for (int32_t i = 0; i < elements; ++i)
    max_error = fmaxf(max_error, fabsf(host[i] - (float)iterations));
  printf("cuda_graph_smoke elements=%d iterations=%d total_ms=%.6f "
         "per_iteration_us=%.6f max_error=%g\n",
         (int)elements, iterations, elapsed,
         1000.0 * elapsed / (double)iterations, (double)max_error);

  polygeist_cublas_destroy();
  cudaFree(x);
  cudaFree(y);
  free(host);
  return max_error == 0.0f ? 0 : 1;
}
