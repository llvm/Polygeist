#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_CUDA_SYNC
extern void bench_cuda_event_init(void);
extern void bench_cuda_event_start(void);
extern double bench_cuda_event_stop_ms(void);
extern void bench_cuda_event_destroy(void);
#endif

__attribute__((noinline))
void gemm_kernel(double *output, const double *left, const double *right,
                 int n, int m, int k) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p)
        sum += left[(size_t)i * k + p] * right[(size_t)p * m + j];
      output[(size_t)i * m + j] = sum;
    }
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec * 1.0e-6;
}

static double sync_backend(void) {
#ifdef USE_CUDA_SYNC
  return bench_cuda_event_stop_ms();
#else
  return -1.0;
#endif
}

static void *page_alloc(size_t bytes) {
  void *ptr = NULL;
  size_t rounded = (bytes + 4095u) & ~(size_t)4095u;
  if (posix_memalign(&ptr, 4096, rounded) != 0)
    return NULL;
  return ptr;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 512;
  int m = argc > 2 ? atoi(argv[2]) : 512;
  int k = argc > 3 ? atoi(argv[3]) : 512;
  size_t left_count = (size_t)n * k;
  size_t right_count = (size_t)k * m;
  size_t output_count = (size_t)n * m;
  double *left = page_alloc(left_count * sizeof(*left));
  double *right = page_alloc(right_count * sizeof(*right));
  double *output = page_alloc(output_count * sizeof(*output));
  if (!left || !right || !output)
    return 2;

  for (size_t i = 0; i < left_count; ++i)
    left[i] = 1.0;
  for (size_t i = 0; i < right_count; ++i)
    right[i] = 1.0;
  for (size_t i = 0; i < output_count; ++i)
    output[i] = -1.0;

#ifdef USE_CUDA_SYNC
  bench_cuda_event_init();
#endif
  for (int iteration = 0; iteration < 10; ++iteration) {
    double begin = now_ms();
#ifdef USE_CUDA_SYNC
    bench_cuda_event_start();
#endif
    gemm_kernel(output, left, right, n, m, k);
    double device_ms = sync_backend();
    double elapsed = now_ms() - begin;
#ifdef USE_CUDA_SYNC
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.9f device_ms=%.9f\n",
           iteration < 5 ? "warmup" : "sample",
           iteration < 5 ? iteration : iteration - 5, elapsed, device_ms);
#else
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.9f\n",
           iteration < 5 ? "warmup" : "sample",
           iteration < 5 ? iteration : iteration - 5, elapsed);
#endif
  }

  double max_abs = 0.0;
  double max_rel = 0.0;
  double expected = (double)k;
  for (size_t i = 0; i < output_count; ++i) {
    double abs_error = fabs(output[i] - expected);
    double rel_error = abs_error / fabs(expected);
    if (abs_error > max_abs)
      max_abs = abs_error;
    if (rel_error > max_rel)
      max_rel = rel_error;
  }
  int pass = max_abs <= 1.0e-9 || max_rel <= 1.0e-9;
  printf("BENCH_CORRECTNESS max_abs=%.17g max_rel=%.17g atol=1e-9 rtol=1e-9\n",
         max_abs, max_rel);
  printf("BENCH_RESULT workload=ginsbach_gemm shape=%dx%dx%d status=%s\n",
         n, m, k, pass ? "PASS" : "FAIL");

#ifdef USE_CUDA_SYNC
  bench_cuda_event_destroy();
#endif
  free(output);
  free(right);
  free(left);
  return pass ? 0 : 1;
}
