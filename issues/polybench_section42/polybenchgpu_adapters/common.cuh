#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void pbgpu_checked(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "native PolyBenchGPU %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::abort();
  }
}

static inline void pbgpu_report(cudaEvent_t start, cudaEvent_t stop) {
  pbgpu_checked(cudaEventRecord(stop), "record stop event");
  pbgpu_checked(cudaEventSynchronize(stop), "synchronize stop event");
  float device_ms = 0.0f;
  pbgpu_checked(cudaEventElapsedTime(&device_ms, start, stop),
                "measure device time");
  std::fprintf(stderr, "POLYBENCH_NATIVE_GPU_TIMING device_ms=%.9f\n",
               static_cast<double>(device_ms));
}
