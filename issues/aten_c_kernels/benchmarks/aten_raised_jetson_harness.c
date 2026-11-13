#include <math.h>
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

#if defined(BENCH_ATEN_CONV2D)
#define OH (H - KH + 1)
#define OW (W - KW + 1)
#elif defined(BENCH_ATEN_MAX_POOL2D)
#define OH ((H - K) / S + 1)
#define OW ((W - K) / S + 1)
#endif

static double seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static double value(size_t i, int salt) {
  const int centered = (int)((i * 17 + (size_t)salt * 13 + 5) % 101) - 50;
  return (double)centered / 257.0;
}

static void fill_f32(float *p, size_t n, int salt) {
  for (size_t i = 0; i < n; ++i) p[i] = (float)value(i, salt);
}

static void fill_f64(double *p, size_t n, int salt) {
  for (size_t i = 0; i < n; ++i) p[i] = value(i, salt);
}

static int compare_f32(const float *a, const float *b, size_t n,
                       double *max_abs, double *max_rel) {
  *max_abs = 0.0;
  *max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)a[i] - (double)b[i]);
    double r = d / fmax(1.0, fabs((double)b[i]));
    *max_abs = fmax(*max_abs, d);
    *max_rel = fmax(*max_rel, r);
  }
#if defined(BENCH_ATEN_RMS_NORM)
  // The extracted scalar C reference accumulates 8M squares strictly in
  // float loop order.  The library implementation uses a tree reduction,
  // whose result is substantially closer to a double-precision sum but is
  // not bitwise-equivalent to that increasingly ill-conditioned ordering.
  return *max_abs <= 2.0e-3 || *max_rel <= 2.0e-3;
#else
  return *max_abs <= 2.0e-4 || *max_rel <= 2.0e-4;
#endif
}

static int compare_f64(const double *a, const double *b, size_t n,
                       double *max_abs, double *max_rel) {
  *max_abs = 0.0;
  *max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs(a[i] - b[i]);
    double r = d / fmax(1.0, fabs(b[i]));
    *max_abs = fmax(*max_abs, d);
    *max_rel = fmax(*max_rel, r);
  }
  return *max_abs <= 1.0e-10 || *max_rel <= 1.0e-10;
}

#if defined(BENCH_ATEN_ADD)
#define BENCH_NAME "aten_add"
extern void aten_add(float *, float *);
extern void aten_add_reference(float *, float *);
static size_t output_size(void) { return (size_t)B*C*H*W; }
static void call_raised(void **p) { aten_add(p[0], p[1]); }
static void call_reference(void **p) { aten_add_reference(p[0], p[1]); }
#elif defined(BENCH_ATEN_ADDMM)
#define BENCH_NAME "aten_addmm"
extern void aten_addmm(double *, double *, double *, double, double);
extern void aten_addmm_reference(double *, double *, double *, double, double);
static size_t output_size(void) { return (size_t)M*N; }
static void call_raised(void **p) { aten_addmm(p[0], p[1], p[2], 0.5, 1.25); }
static void call_reference(void **p) { aten_addmm_reference(p[0], p[1], p[2], 0.5, 1.25); }
#elif defined(BENCH_ATEN_BATCH_NORM)
#define BENCH_NAME "aten_batch_norm"
extern void aten_batch_norm(float *, float *, float *, float *, float *, float *);
extern void aten_batch_norm_reference(float *, float *, float *, float *, float *, float *);
static size_t output_size(void) { return (size_t)B*C*H*W; }
static void call_raised(void **p) { aten_batch_norm(p[0],p[1],p[2],p[3],p[4],p[5]); }
static void call_reference(void **p) { aten_batch_norm_reference(p[0],p[1],p[2],p[3],p[4],p[5]); }
#elif defined(BENCH_ATEN_CONV2D)
#define BENCH_NAME "aten_conv2d"
extern void aten_conv2d(float *, float *, float *);
extern void aten_conv2d_reference(float *, float *, float *);
static size_t output_size(void) { return (size_t)B*OC*OH*OW; }
static void call_raised(void **p) { aten_conv2d(p[0], p[1], p[2]); }
static void call_reference(void **p) { aten_conv2d_reference(p[0], p[1], p[2]); }
#elif defined(BENCH_ATEN_DOT)
#define BENCH_NAME "aten_dot"
extern void aten_dot(double *, double *, double *);
extern void aten_dot_reference(double *, double *, double *);
static size_t output_size(void) { return 1; }
static void call_raised(void **p) { aten_dot(p[0], p[1], p[2]); }
static void call_reference(void **p) { aten_dot_reference(p[0], p[1], p[2]); }
#elif defined(BENCH_ATEN_GELU)
#define BENCH_NAME "aten_gelu"
extern void aten_gelu(float *, float *);
extern void aten_gelu_reference(float *, float *);
static size_t output_size(void) { return N; }
static void call_raised(void **p) { aten_gelu(p[0], p[1]); }
static void call_reference(void **p) { aten_gelu_reference(p[0], p[1]); }
#elif defined(BENCH_ATEN_MAX_POOL2D)
#define BENCH_NAME "aten_max_pool2d"
extern void aten_max_pool2d(float *, float *);
extern void aten_max_pool2d_reference(float *, float *);
static size_t output_size(void) { return (size_t)B*C*OH*OW; }
static void call_raised(void **p) { aten_max_pool2d(p[0], p[1]); }
static void call_reference(void **p) { aten_max_pool2d_reference(p[0], p[1]); }
#elif defined(BENCH_ATEN_MM)
#define BENCH_NAME "aten_mm"
extern void aten_mm(double *, double *, double *);
extern void aten_mm_reference(double *, double *, double *);
static size_t output_size(void) { return (size_t)M*N; }
static void call_raised(void **p) { aten_mm(p[0], p[1], p[2]); }
static void call_reference(void **p) { aten_mm_reference(p[0], p[1], p[2]); }
#elif defined(BENCH_ATEN_MV)
#define BENCH_NAME "aten_mv"
extern void aten_mv(double *, double *, double *);
extern void aten_mv_reference(double *, double *, double *);
static size_t output_size(void) { return M; }
static void call_raised(void **p) { aten_mv(p[0], p[1], p[2]); }
static void call_reference(void **p) { aten_mv_reference(p[0], p[1], p[2]); }
#elif defined(BENCH_ATEN_RMS_NORM)
#define BENCH_NAME "aten_rms_norm"
extern void aten_rms_norm(float *, float *, float *, float);
extern void aten_rms_norm_reference(float *, float *, float *, float);
static size_t output_size(void) { return N; }
static void call_raised(void **p) { aten_rms_norm(p[0], p[1], p[2], 1.0e-5f); }
static void call_reference(void **p) { aten_rms_norm_reference(p[0], p[1], p[2], 1.0e-5f); }
#elif defined(BENCH_ATEN_SOFTMAX)
#define BENCH_NAME "aten_softmax"
extern void aten_softmax(float *);
extern void aten_softmax_reference(float *);
static size_t output_size(void) { return N; }
static void call_raised(void **p) { aten_softmax(p[0]); }
static void call_reference(void **p) { aten_softmax_reference(p[0]); }
#else
#error "Select one BENCH_ATEN_* kernel"
#endif

int main(void) {
  void *raised[6] = {0};
  void *reference[6] = {0};
  size_t sizes[6] = {0};
  int output_index = 0;
  int is_f64 = 0;

#if defined(BENCH_ATEN_ADD)
  sizes[0]=sizes[1]=(size_t)B*C*H*W; output_index=1;
#elif defined(BENCH_ATEN_ADDMM)
  sizes[0]=(size_t)M*K; sizes[1]=(size_t)K*N; sizes[2]=(size_t)M*N; output_index=2; is_f64=1;
#elif defined(BENCH_ATEN_BATCH_NORM)
  sizes[0]=sizes[5]=(size_t)B*C*H*W; sizes[1]=sizes[2]=sizes[3]=sizes[4]=C; output_index=5;
#elif defined(BENCH_ATEN_CONV2D)
  sizes[0]=(size_t)B*IC*H*W; sizes[1]=(size_t)OC*IC*KH*KW; sizes[2]=(size_t)B*OC*OH*OW; output_index=2;
#elif defined(BENCH_ATEN_DOT)
  sizes[0]=sizes[1]=N; sizes[2]=1; output_index=2; is_f64=1;
#elif defined(BENCH_ATEN_GELU)
  sizes[0]=sizes[1]=N; output_index=1;
#elif defined(BENCH_ATEN_MAX_POOL2D)
  sizes[0]=(size_t)B*C*H*W; sizes[1]=(size_t)B*C*OH*OW; output_index=1;
#elif defined(BENCH_ATEN_MM)
  sizes[0]=(size_t)M*K; sizes[1]=(size_t)K*N; sizes[2]=(size_t)M*N; output_index=2; is_f64=1;
#elif defined(BENCH_ATEN_MV)
  sizes[0]=(size_t)M*K; sizes[1]=K; sizes[2]=M; output_index=2; is_f64=1;
#elif defined(BENCH_ATEN_RMS_NORM)
  sizes[0]=sizes[1]=sizes[2]=N; output_index=2;
#elif defined(BENCH_ATEN_SOFTMAX)
  sizes[0]=N; output_index=0;
#endif

  for (int p = 0; p < 6; ++p) {
    if (!sizes[p]) continue;
    size_t bytes = sizes[p] * (is_f64 ? sizeof(double) : sizeof(float));
#ifdef DEVICE_RESIDENT
    CUDA_CHECK(cudaMalloc(&raised[p], bytes));
#else
    raised[p] = malloc(bytes);
#endif
    reference[p] = malloc(bytes);
    if (!raised[p] || !reference[p]) return 2;
#ifdef DEVICE_RESIDENT
    if (is_f64) fill_f64(reference[p], sizes[p], p + 1);
    else fill_f32(reference[p], sizes[p], p + 1);
    CUDA_CHECK(cudaMemcpy(raised[p], reference[p], bytes,
                          cudaMemcpyHostToDevice));
#else
    if (is_f64) fill_f64(raised[p], sizes[p], p + 1);
    else fill_f32(raised[p], sizes[p], p + 1);
    memcpy(reference[p], raised[p], bytes);
#endif
  }

#if defined(BENCH_ATEN_BATCH_NORM)
  // inv_std is mathematically positive in inference.  cuDNN accepts variance
  // and epsilon, so use valid inverse standard deviations that can be
  // reconstructed without losing a sign.
  for (size_t c = 0; c < C; ++c) {
    ((float *)reference[3])[c] = 0.75f + 0.005f * (float)(c % 41);
#ifndef DEVICE_RESIDENT
    ((float *)raised[3])[c] = ((float *)reference[3])[c];
#endif
  }
#ifdef DEVICE_RESIDENT
  CUDA_CHECK(cudaMemcpy(raised[3], reference[3], C * sizeof(float),
                        cudaMemcpyHostToDevice));
#endif
#endif

  call_reference(reference);
  call_raised(raised);
  double max_abs, max_rel;
#ifdef DEVICE_RESIDENT
  void *actual = malloc(output_size() * (is_f64 ? sizeof(double) : sizeof(float)));
  if (!actual) return 2;
  CUDA_CHECK(cudaMemcpy(actual, raised[output_index],
                        output_size() * (is_f64 ? sizeof(double) : sizeof(float)),
                        cudaMemcpyDeviceToHost));
#else
  void *actual = raised[output_index];
#endif
  int correct = is_f64
      ? compare_f64(actual, reference[output_index],
                    output_size(), &max_abs, &max_rel)
      : compare_f32(actual, reference[output_index],
                    output_size(), &max_abs, &max_rel);
#ifdef DEVICE_RESIDENT
  free(actual);
#endif
  printf("kernel=%s correctness=%s max_abs=%.17g max_rel=%.17g\n",
         BENCH_NAME, correct ? "PASS" : "FAIL", max_abs, max_rel);
  if (!correct) return 1;

  call_raised(raised);
  const double start = seconds();
  for (int i = 0; i < BENCH_ITERS; ++i) call_raised(raised);
  const double runtime_us = (seconds() - start) * 1.0e6 / BENCH_ITERS;
#ifdef DEVICE_RESIDENT
  printf("kernel=%s mode=raised_device iterations=%d raised_device_us=%.6f\n",
         BENCH_NAME, BENCH_ITERS, runtime_us);
#else
  printf("kernel=%s iterations=%d raised_gpu_us=%.6f\n",
         BENCH_NAME, BENCH_ITERS, runtime_us);
#endif
  for (int p = 0; p < 6; ++p) {
#ifdef DEVICE_RESIDENT
    if (raised[p]) CUDA_CHECK(cudaFree(raised[p]));
#else
    free(raised[p]);
#endif
    free(reference[p]);
  }
  return 0;
}
