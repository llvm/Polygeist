#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef POLYGEIST_BENCH_CUDA
#include <cuda_runtime_api.h>
#endif

void adds2m1(double *a, double *b, double coefficient, int count);

enum { kWarmups = 5, kSamples = 5, kCount = 1048576 };

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
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
  const size_t bytes = (size_t)kCount * sizeof(double);
  const double coefficient = -0.375;
  double *initial_a = malloc(bytes), *host_b = malloc(bytes);
  double *actual = malloc(bytes);
  if (!initial_a || !host_b || !actual)
    return 100;
  for (int i = 0; i < kCount; ++i) {
    initial_a[i] = ((i % 31) - 15) * 0.0625;
    host_b[i] = ((i % 23) - 11) * 0.03125;
  }

  double *a = actual, *b = host_b;
#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMalloc((void **)&a, bytes));
  CUDA_OK(cudaMalloc((void **)&b, bytes));
  CUDA_OK(cudaMemcpy(b, host_b, bytes, cudaMemcpyHostToDevice));
#endif
  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaMemcpy(a, initial_a, bytes, cudaMemcpyHostToDevice));
#else
    memcpy(a, initial_a, bytes);
#endif
    double begin = wall_ms();
    adds2m1(a, b, coefficient, kCount);
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaDeviceSynchronize());
#endif
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }
#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMemcpy(actual, a, bytes, cudaMemcpyDeviceToHost));
#endif

  double max_abs = 0.0;
  int ok = 1;
  for (int i = 0; i < kCount; ++i) {
    double expected = initial_a[i] + coefficient * host_b[i];
    double error = fabs(actual[i] - expected);
    if (!isfinite(actual[i]))
      ok = 0;
    if (error > max_abs)
      max_abs = error;
  }
  ok = ok && max_abs <= 1.0e-12;
  printf("BENCH_CORRECTNESS max_abs=%.17g atol=1e-12\n", max_abs);
  printf("BENCH_RESULT workload=npb_ua_daxpby n=%d status=%s\n", kCount,
         ok ? "PASS" : "FAIL");
#ifdef POLYGEIST_BENCH_CUDA
  cudaFree(a);
  cudaFree(b);
#endif
  free(initial_a);
  free(host_b);
  free(actual);
  return ok ? 0 : 1;
}
