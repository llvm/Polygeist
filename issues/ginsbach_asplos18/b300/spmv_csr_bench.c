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
void spmv_csr_kernel(double *output, const double *matrix,
                     const double *vector, const int *ranges,
                     const int *indir, int rows) {
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = ranges[i]; j < ranges[i + 1]; ++j)
      sum += matrix[j] * vector[indir[j]];
    output[i] = sum;
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
  int rows = argc > 1 ? atoi(argv[1]) : 1000000;
  int cols = argc > 2 ? atoi(argv[2]) : 1000000;
  int entries_per_row = argc > 3 ? atoi(argv[3]) : 27;
  size_t nnz = (size_t)rows * entries_per_row;
  double *matrix = page_alloc(nnz * sizeof(*matrix));
  double *vector = page_alloc((size_t)cols * sizeof(*vector));
  double *output = page_alloc((size_t)rows * sizeof(*output));
  int *ranges = page_alloc(((size_t)rows + 1) * sizeof(*ranges));
  int *indir = page_alloc(nnz * sizeof(*indir));
  if (!matrix || !vector || !output || !ranges || !indir)
    return 2;

  for (int i = 0; i <= rows; ++i)
    ranges[i] = i * entries_per_row;
  for (size_t j = 0; j < nnz; ++j) {
    matrix[j] = 1.0;
    indir[j] = (int)((j * 17u + j / (size_t)entries_per_row) % (size_t)cols);
  }
  for (int i = 0; i < cols; ++i)
    vector[i] = 1.0;
  for (int i = 0; i < rows; ++i)
    output[i] = -1.0;

#ifdef USE_CUDA_SYNC
  bench_cuda_event_init();
#endif
  for (int iteration = 0; iteration < 10; ++iteration) {
    double begin = now_ms();
#ifdef USE_CUDA_SYNC
    bench_cuda_event_start();
#endif
    spmv_csr_kernel(output, matrix, vector, ranges, indir, rows);
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
  double expected = (double)entries_per_row;
  for (int i = 0; i < rows; ++i) {
    double abs_error = fabs(output[i] - expected);
    double rel_error = abs_error / fabs(expected);
    if (abs_error > max_abs)
      max_abs = abs_error;
    if (rel_error > max_rel)
      max_rel = rel_error;
  }
  int pass = max_abs <= 1.0e-12 || max_rel <= 1.0e-12;
  printf("BENCH_CORRECTNESS max_abs=%.17g max_rel=%.17g atol=1e-12 rtol=1e-12\n",
         max_abs, max_rel);
  printf("BENCH_RESULT workload=ginsbach_spmv_csr shape=%dx%d_nnz=%zu status=%s\n",
         rows, cols, nnz, pass ? "PASS" : "FAIL");

#ifdef USE_CUDA_SYNC
  bench_cuda_event_destroy();
#endif
  free(indir);
  free(ranges);
  free(output);
  free(vector);
  free(matrix);
  return pass ? 0 : 1;
}
