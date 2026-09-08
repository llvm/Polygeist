#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef POLYGEIST_BENCH_CUDA
#include <cuda_runtime_api.h>
#endif

void cpu_stencil(float c0, float c1, float *input, float *output, int nx,
                 int ny, int nz);

enum { kWarmups = 5, kSamples = 5, kNx = 128, kNy = 128, kNz = 128 };

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

static size_t index3d(int i, int j, int k) {
  return (size_t)i + kNx * ((size_t)j + kNy * (size_t)k);
}

#ifdef POLYGEIST_BENCH_CUDA
#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(status));       \
      return 2;                                                                \
    }                                                                          \
  } while (0)
#endif

int main(void) {
  const size_t count = (size_t)kNx * kNy * kNz;
  const size_t bytes = count * sizeof(float);
  const float c0 = 1.0f / 6.0f, c1 = 1.0f / 36.0f;
  float *host_input = malloc(bytes), *actual = malloc(bytes);
  if (!host_input || !actual)
    return 100;
  for (size_t i = 0; i < count; ++i)
    host_input[i] = ((int)(i % 29) - 14) * 0.015625f;

  float *input = host_input, *output = actual;
#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMalloc((void **)&input, bytes));
  CUDA_OK(cudaMalloc((void **)&output, bytes));
  CUDA_OK(cudaMemcpy(input, host_input, bytes, cudaMemcpyHostToDevice));
#endif

  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaMemset(output, 0, bytes));
#else
    memset(output, 0, bytes);
#endif
    double begin = wall_ms();
    cpu_stencil(c0, c1, input, output, kNx, kNy, kNz);
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaDeviceSynchronize());
#endif
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }

#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMemcpy(actual, output, bytes, cudaMemcpyDeviceToHost));
#endif
  float max_abs = 0.0f, max_rel = 0.0f;
  int ok = 1;
  for (int k = 0; k < kNz; ++k) {
    for (int j = 0; j < kNy; ++j) {
      for (int i = 0; i < kNx; ++i) {
        size_t p = index3d(i, j, k);
        float expected = 0.0f;
        if (i > 0 && i + 1 < kNx && j > 0 && j + 1 < kNy && k > 0 &&
            k + 1 < kNz) {
          expected = c1 * (host_input[index3d(i + 1, j, k)] +
                           host_input[index3d(i - 1, j, k)] +
                           host_input[index3d(i, j + 1, k)] +
                           host_input[index3d(i, j - 1, k)] +
                           host_input[index3d(i, j, k + 1)] +
                           host_input[index3d(i, j, k - 1)]) -
                     c0 * host_input[p];
        }
        float absolute = fabsf(actual[p] - expected);
        float relative = absolute / fmaxf(1.0f, fabsf(expected));
        if (!isfinite(actual[p]))
          ok = 0;
        if (absolute > max_abs)
          max_abs = absolute;
        if (relative > max_rel)
          max_rel = relative;
      }
    }
  }
  ok = ok && (max_abs <= 1.0e-5f || max_rel <= 1.0e-5f);
  printf("BENCH_CORRECTNESS max_abs=%g max_rel=%g atol=1e-5 rtol=1e-5\n",
         max_abs, max_rel);
  printf("BENCH_RESULT workload=parboil_stencil shape=128x128x128 status=%s\n",
         ok ? "PASS" : "FAIL");

#ifdef POLYGEIST_BENCH_CUDA
  cudaFree(input);
  cudaFree(output);
#endif
  free(host_input);
  free(actual);
  return ok ? 0 : 1;
}
