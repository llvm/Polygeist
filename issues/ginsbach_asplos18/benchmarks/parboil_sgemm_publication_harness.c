#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef POLYGEIST_BENCH_CUDA
#include <cuda_runtime_api.h>
#endif

void parboil_basic_sgemm(char transa, char transb, int m, int n, int k,
                         float alpha, const float *a, int lda,
                         const float *b, int ldb, float beta, float *c,
                         int ldc);

enum { kWarmups = 5, kSamples = 5, kM = 512, kN = 512, kK = 512 };

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

static void fill_inputs(float *a, float *b, float *c) {
  for (size_t i = 0; i < (size_t)kM * kK; ++i)
    a[i] = ((int)(i % 17) - 8) * 0.03125f;
  for (size_t i = 0; i < (size_t)kN * kK; ++i)
    b[i] = ((int)(i % 13) - 6) * 0.0625f;
  for (size_t i = 0; i < (size_t)kM * kN; ++i)
    c[i] = ((int)(i % 7) - 3) * 0.125f;
}

static int validate(const float *a, const float *b, const float *initial_c,
                    const float *actual) {
  const float alpha = 1.25f, beta = -0.5f;
  float max_abs = 0.0f, max_rel = 0.0f;
  for (int n = 0; n < kN; ++n) {
    for (int m = 0; m < kM; ++m) {
      double dot = 0.0;
      for (int k = 0; k < kK; ++k)
        dot += (double)a[m + k * kM] * b[n + k * kN];
      float expected = beta * initial_c[m + n * kM] + alpha * (float)dot;
      float absolute = fabsf(actual[m + n * kM] - expected);
      float relative = absolute / fmaxf(1.0f, fabsf(expected));
      if (!isfinite(actual[m + n * kM]))
        return 0;
      if (absolute > max_abs)
        max_abs = absolute;
      if (relative > max_rel)
        max_rel = relative;
    }
  }
  printf("BENCH_CORRECTNESS max_abs=%g max_rel=%g atol=1e-4 rtol=1e-4\n",
         max_abs, max_rel);
  return max_abs <= 1.0e-4f || max_rel <= 1.0e-4f;
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
  const size_t a_bytes = (size_t)kM * kK * sizeof(float);
  const size_t b_bytes = (size_t)kN * kK * sizeof(float);
  const size_t c_bytes = (size_t)kM * kN * sizeof(float);
  float *host_a = malloc(a_bytes), *host_b = malloc(b_bytes);
  float *initial_c = malloc(c_bytes), *actual = malloc(c_bytes);
  if (!host_a || !host_b || !initial_c || !actual)
    return 100;
  fill_inputs(host_a, host_b, initial_c);

  float *a = host_a, *b = host_b, *c = actual;
#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMalloc((void **)&a, a_bytes));
  CUDA_OK(cudaMalloc((void **)&b, b_bytes));
  CUDA_OK(cudaMalloc((void **)&c, c_bytes));
  CUDA_OK(cudaMemcpy(a, host_a, a_bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(b, host_b, b_bytes, cudaMemcpyHostToDevice));
#endif

  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaMemcpy(c, initial_c, c_bytes, cudaMemcpyHostToDevice));
#else
    memcpy(c, initial_c, c_bytes);
#endif
    double begin = wall_ms();
    parboil_basic_sgemm('N', 'T', kM, kN, kK, 1.25f, a, kM, b, kN,
                        -0.5f, c, kM);
#ifdef POLYGEIST_BENCH_CUDA
    CUDA_OK(cudaDeviceSynchronize());
#endif
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }

#ifdef POLYGEIST_BENCH_CUDA
  CUDA_OK(cudaMemcpy(actual, c, c_bytes, cudaMemcpyDeviceToHost));
#endif
  int ok = validate(host_a, host_b, initial_c, actual);
  printf("BENCH_RESULT workload=parboil_sgemm shape=512x512x512 status=%s\n",
         ok ? "PASS" : "FAIL");

#ifdef POLYGEIST_BENCH_CUDA
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
#endif
  free(host_a);
  free(host_b);
  free(initial_c);
  free(actual);
  return ok ? 0 : 1;
}
