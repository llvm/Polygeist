/* Correctness-gated warm timing harness for the exhaustive ATen FULL/FULL
 * batch.  FUNCTION and REFERENCE name the raised and -O3 reference symbols;
 * one BENCH_KIND_* macro selects their common signature and allocation shape.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef DEVICE_RESIDENT
#include <cuda_runtime_api.h>
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t status_ = (expr);                                             \
    if (status_ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(status_));                                   \
      return 2;                                                              \
    }                                                                        \
  } while (0)
#endif

#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#define STR1(x) #x
#define STR(x) STR1(x)

static double seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}
static void fill_f32(float *p, size_t n, int salt) {
  for (size_t i = 0; i < n; ++i)
    p[i] = (float)((int)((i * 17 + (size_t)salt * 13 + 5) % 101) - 50) / 257.0f;
}
static void fill_f64(double *p, size_t n, int salt) {
  for (size_t i = 0; i < n; ++i)
    p[i] = (double)((int)((i * 17 + (size_t)salt * 13 + 5) % 101) - 50) / 257.0;
}
static int compare_f32(const float *a, const float *b, size_t n,
                       double *max_abs, double *max_rel) {
  *max_abs = *max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)a[i] - (double)b[i]);
    double r = d / fmax(1.0, fabs((double)b[i]));
    *max_abs = fmax(*max_abs, d); *max_rel = fmax(*max_rel, r);
  }
#if defined(BENCH_KIND_CONV3D_BIAS) || defined(BENCH_KIND_CONV3D) || \
    defined(BENCH_KIND_CONV3D_TRANSPOSE_BACKWARD)
  return *max_abs <= 5.0e-4 || *max_rel <= 5.0e-4;
#else
  return *max_abs <= 2.0e-4 || *max_rel <= 2.0e-4;
#endif
}
static int compare_f64(const double *a, const double *b, size_t n,
                       double *max_abs, double *max_rel) {
  *max_abs = *max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs(a[i] - b[i]);
    double r = d / fmax(1.0, fabs(b[i]));
    *max_abs = fmax(*max_abs, d); *max_rel = fmax(*max_rel, r);
  }
  return *max_abs <= 1.0e-9 || *max_rel <= 1.0e-9;
}

#if defined(BENCH_KIND_COPY1)
extern void FUNCTION(float *, float *); extern void REFERENCE(float *, float *);
static const size_t sizes[] = {N, N}; static const int outputs[] = {1};
static void call_raised(void **p) { FUNCTION(p[0], p[1]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1]); }
#elif defined(BENCH_KIND_COPY2)
extern void FUNCTION(float *, float *); extern void REFERENCE(float *, float *);
static const size_t sizes[] = {(size_t)B*N, (size_t)B*N}; static const int outputs[] = {1};
static void call_raised(void **p) { FUNCTION(p[0], p[1]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1]); }
#elif defined(BENCH_KIND_TWO_COPY)
extern void FUNCTION(float *, float *, float *, float *);
extern void REFERENCE(float *, float *, float *, float *);
static const size_t sizes[] = {N, N, N, N}; static const int outputs[] = {2, 3};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2], p[3]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2], p[3]); }
#elif defined(BENCH_KIND_AS_COMPLEX)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)N*2, N, N}; static const int outputs[] = {1, 2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_CAT)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)R*K, (size_t)M*K, (size_t)(R+M)*K};
static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_NARROW)
extern void FUNCTION(float *, float *); extern void REFERENCE(float *, float *);
static const size_t sizes[] = {(size_t)R*C, (size_t)R*L}; static const int outputs[] = {1};
static void call_raised(void **p) { FUNCTION(p[0], p[1]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1]); }
#elif defined(BENCH_KIND_DOT)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {K, K, 1}; static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_GEMV)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
#ifdef GEMV_TRANS
static const size_t sizes[] = {(size_t)M*K, M, K};
#else
static const size_t sizes[] = {(size_t)M*K, K, M};
#endif
static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_BATCH_GEMM)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)B*M*K, (size_t)K*N, (size_t)B*M*N};
static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_OUTER)
extern void FUNCTION(double *, double *, double *); extern void REFERENCE(double *, double *, double *);
static const size_t sizes[] = {M, N, (size_t)M*N}; static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#define IS_F64 1
#elif defined(BENCH_KIND_GELU)
extern void FUNCTION(float *, float *); extern void REFERENCE(float *, float *);
static const size_t sizes[] = {N, N}; static const int outputs[] = {1};
static void call_raised(void **p) { FUNCTION(p[0], p[1]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1]); }
#elif defined(BENCH_KIND_LINEAR_COMB)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)4*N, 4, N}; static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_ZERO)
extern void FUNCTION(float *); extern void REFERENCE(float *);
static const size_t sizes[] = {N}; static const int outputs[] = {0};
static void call_raised(void **p) { FUNCTION(p[0]); }
static void call_reference(void **p) { REFERENCE(p[0]); }
#elif defined(BENCH_KIND_CONV3D_BIAS)
extern void FUNCTION(float *, float *, float *, float *);
extern void REFERENCE(float *, float *, float *, float *);
static const size_t sizes[] = {(size_t)B*IC*D*H*W, (size_t)OC*IC*K*K*K, OC,
                               (size_t)B*OC*(D-K+1)*(H-K+1)*(W-K+1)};
static const int outputs[] = {3};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2], p[3]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2], p[3]); }
#elif defined(BENCH_KIND_CONV3D)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)C*D*H*W, (size_t)O*C*K*K*K,
                               (size_t)O*(D-K+1)*(H-K+1)*(W-K+1)};
static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#elif defined(BENCH_KIND_CONV3D_TRANSPOSE_BACKWARD)
extern void FUNCTION(float *, float *, float *); extern void REFERENCE(float *, float *, float *);
static const size_t sizes[] = {(size_t)O*(D+2)*(H+2)*(W+2), (size_t)C*O*K*K*K,
                               (size_t)C*D*H*W};
static const int outputs[] = {2};
static void call_raised(void **p) { FUNCTION(p[0], p[1], p[2]); }
static void call_reference(void **p) { REFERENCE(p[0], p[1], p[2]); }
#else
#error Select BENCH_KIND_*
#endif

int main(void) {
  enum { ARGC = sizeof(sizes) / sizeof(sizes[0]),
         NOUT = sizeof(outputs) / sizeof(outputs[0]) };
  void *raised[4] = {0}, *reference[4] = {0};
#ifdef IS_F64
  const size_t elem = sizeof(double);
#else
  const size_t elem = sizeof(float);
#endif
  for (int i = 0; i < ARGC; ++i) {
#ifdef DEVICE_RESIDENT
    CUDA_CHECK(cudaMalloc(&raised[i], sizes[i] * elem));
#else
    raised[i] = malloc(sizes[i] * elem);
#endif
    reference[i] = malloc(sizes[i] * elem);
    if (!raised[i] || !reference[i]) return 2;
#ifdef IS_F64
#ifdef DEVICE_RESIDENT
    fill_f64(reference[i], sizes[i], i + 1);
#else
    fill_f64(raised[i], sizes[i], i + 1);
#endif
#else
#ifdef DEVICE_RESIDENT
    fill_f32(reference[i], sizes[i], i + 1);
#else
    fill_f32(raised[i], sizes[i], i + 1);
#endif
#endif
#ifdef DEVICE_RESIDENT
    CUDA_CHECK(cudaMemcpy(raised[i], reference[i], sizes[i] * elem,
                          cudaMemcpyHostToDevice));
#else
    memcpy(reference[i], raised[i], sizes[i] * elem);
#endif
  }
#ifdef BENCH_KIND_DOT
  /* Avoid a cancellation-dominated condition number when comparing the
   * sequential C reduction with cuBLAS's tree reduction at 16M elements. */
  for (int p = 0; p < 2; ++p) {
#ifdef DEVICE_RESIDENT
    float *x = (float *)reference[p];
#else
    float *x = (float *)raised[p];
#endif
    for (size_t i = 0; i < sizes[p]; ++i)
      x[i] = (i & 1) ? -1.0f : 1.0f;
#ifdef DEVICE_RESIDENT
    CUDA_CHECK(cudaMemcpy(raised[p], reference[p],
                          sizes[p] * sizeof(float), cudaMemcpyHostToDevice));
#else
    memcpy(reference[p], raised[p], sizes[p] * sizeof(float));
#endif
  }
#endif
  call_reference(reference); call_raised(raised);
  int correct = 1; double max_abs = 0.0, max_rel = 0.0;
  for (int j = 0; j < NOUT; ++j) {
    int i = outputs[j]; double a = 0.0, r = 0.0;
#ifdef DEVICE_RESIDENT
    void *actual = malloc(sizes[i] * elem);
    if (!actual) return 2;
    CUDA_CHECK(cudaMemcpy(actual, raised[i], sizes[i] * elem,
                          cudaMemcpyDeviceToHost));
#else
    void *actual = raised[i];
#endif
#ifdef IS_F64
    correct &= compare_f64(actual, reference[i], sizes[i], &a, &r);
#else
    correct &= compare_f32(actual, reference[i], sizes[i], &a, &r);
#endif
#ifdef DEVICE_RESIDENT
    free(actual);
#endif
    max_abs = fmax(max_abs, a); max_rel = fmax(max_rel, r);
  }
  printf("kernel=%s correctness=%s max_abs=%.17g max_rel=%.17g\n",
         STR(FUNCTION), correct ? "PASS" : "FAIL", max_abs, max_rel);
  if (!correct) return 1;
  call_raised(raised);
  double start = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) call_raised(raised);
  double us = (seconds() - start) * 1.0e6 / BENCH_ITERS;
#ifdef DEVICE_RESIDENT
  printf("kernel=%s mode=raised_device iterations=%d raised_device_us=%.6f\n",
         STR(FUNCTION), BENCH_ITERS, us);
#else
  printf("kernel=%s iterations=%d raised_gpu_us=%.6f\n",
         STR(FUNCTION), BENCH_ITERS, us);
#endif
  for (int i = 0; i < ARGC; ++i) {
#ifdef DEVICE_RESIDENT
    CUDA_CHECK(cudaFree(raised[i]));
#else
    free(raised[i]);
#endif
    free(reference[i]);
  }
  return 0;
}
