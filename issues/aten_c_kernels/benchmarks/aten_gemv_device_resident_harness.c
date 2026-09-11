/* Correctness and warm timing for the generated ATen GEMV function with
 * cudaMalloc operands.  This intentionally calls FUNCTION, not cuBLAS
 * directly, so raising, matching, ABI lowering, the zero stage, and the
 * cuBLAS shim are all exercised exactly as in the mapped-host harness.
 */
#define _POSIX_C_SOURCE 200809L
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

#define STR1(x) #x
#define STR(x) STR1(x)
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t status_ = (expr);                                             \
    if (status_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(status_));                                   \
      exit(2);                                                               \
    }                                                                        \
  } while (0)

extern void FUNCTION(float *, float *, float *);
extern void REFERENCE(float *, float *, float *);

static double seconds(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void fill(float *values, size_t count, int salt) {
  for (size_t i = 0; i < count; ++i)
    values[i] =
        (float)((int)((i * 17 + (size_t)salt * 13 + 5) % 101) - 50) /
        257.0f;
}

int main(void) {
  const size_t matrix_elems = (size_t)M * K;
#ifdef GEMV_TRANS
  const size_t x_elems = M;
  const size_t y_elems = K;
#else
  const size_t x_elems = K;
  const size_t y_elems = M;
#endif
  float *matrix = (float *)malloc(matrix_elems * sizeof(float));
  float *x = (float *)malloc(x_elems * sizeof(float));
  float *expected = (float *)malloc(y_elems * sizeof(float));
  float *actual = (float *)malloc(y_elems * sizeof(float));
  if (!matrix || !x || !expected || !actual)
    return 2;
  fill(matrix, matrix_elems, 1);
  fill(x, x_elems, 2);
  REFERENCE(matrix, x, expected);

  float *device_matrix = NULL;
  float *device_x = NULL;
  float *device_y = NULL;
  CUDA_CHECK(cudaMalloc((void **)&device_matrix,
                        matrix_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&device_x, x_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&device_y, y_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_matrix, matrix,
                        matrix_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_x, x, x_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  FUNCTION(device_matrix, device_x, device_y);
  CUDA_CHECK(cudaMemcpy(actual, device_y, y_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < y_elems; ++i) {
    double diff = fabs((double)actual[i] - (double)expected[i]);
    double relative = diff / fmax(1.0, fabs((double)expected[i]));
    max_abs = fmax(max_abs, diff);
    max_rel = fmax(max_rel, relative);
  }
  const int correct = max_abs <= 1.0e-3 || max_rel <= 1.0e-3;
  printf("kernel=%s mode=raised_device correctness=%s "
         "max_abs=%.17g max_rel=%.17g\n",
         STR(FUNCTION), correct ? "PASS" : "FAIL", max_abs, max_rel);
  if (!correct)
    return 1;

  FUNCTION(device_matrix, device_x, device_y);
  double start = seconds();
  for (int iteration = 0; iteration < BENCH_ITERS; ++iteration)
    FUNCTION(device_matrix, device_x, device_y);
  double runtime_us = (seconds() - start) * 1.0e6 / BENCH_ITERS;
  printf("kernel=%s mode=raised_device iterations=%d "
         "raised_device_us=%.6f\n",
         STR(FUNCTION), BENCH_ITERS, runtime_us);

  CUDA_CHECK(cudaFree(device_matrix));
  CUDA_CHECK(cudaFree(device_x));
  CUDA_CHECK(cudaFree(device_y));
  free(matrix);
  free(x);
  free(expected);
  free(actual);
  return 0;
}
