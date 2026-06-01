// polygeist_cublas_rt_cuda.c — real cuBLAS implementation of the runtime
// shim ABI. Compile with nvcc (or clang+CUDA) and link against -lcublas
// -lcudart. Build with:
//   nvcc -O3 -c polygeist_cublas_rt_cuda.c -o polygeist_cublas_rt.o
// or, treating the file as C with the cuda toolkit headers in scope:
//   clang -O3 -I${CUDA}/include -c polygeist_cublas_rt_cuda.c -o ...
//
// MEMORY MODEL (Jetson zero-copy via cudaHostRegister):
//   The integrated GPU on Jetson shares physical DRAM with the CPU.
//   Instead of cudaMalloc + cudaMemcpyH2D + cuBLAS + cudaMemcpyD2H + cudaFree
//   (which moves bytes within the same DRAM, pure waste), we cudaHostRegister
//   the polybench-allocated buffers with `cudaHostRegisterMapped`, pass the
//   host pointers directly to cuBLAS via cudaHostGetDevicePointer, then
//   cudaHostUnregister at the end. On a Tegra SoC with UVA, the host and
//   device addresses are the same; the register call only sets up the GPU
//   page-table mapping.
//
//   Aliased operands (e.g. syrk's A passed as both A and B) are handled by
//   the helper register_host_safe() — it ignores
//   cudaErrorHostMemoryAlreadyRegistered so the same pointer can be
//   "registered" multiple times within a single call.
//
// ROW→COL-MAJOR:
//   cuBLAS expects column-major; our linalg.generic is row-major. We compute
//   Cᵀ = α(BᵀAᵀ) + βCᵀ by swapping the A and B operands in the cublasDgemm
//   call (with both transA and transB set to CUBLAS_OP_N). The math is
//   identical, no actual data transpose needed.

#include "polygeist_cublas_rt.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Intentionally do NOT include <cuda_fp16.h> or <cuda_bf16.h>. Those
 * headers use NVCC-specific `__device__` builtins that fail to parse under
 * aarch64-linux-gnu-gcc (our cross-compile path). cuDNN's API is type-agnostic
 * on the data side — it reads the buffer layout from the descriptor
 * (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16 / etc.), so we use uint16_t* for
 * the device buffers in the half-precision paths instead of __half /
 * __nv_bfloat16. Bits are identical, so memcpy from the host's _Float16 /
 * __bf16 arrays via uint16_t lands the correct values on the device. */

static cublasHandle_t   g_handle;
static cublasLtHandle_t g_lt = NULL;
static cudnnHandle_t    g_cudnn = NULL;
static cudaStream_t     g_stream;
static cudaEvent_t    g_ev_begin;
static cudaEvent_t    g_ev_end;
static int            g_initialized = 0;
static int            g_timing_enabled = -1;
static FILE          *g_timing_file = NULL;

#define CUDA_CHECK(call) do {                                                \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "%s:%d cuda error: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call) do {                                              \
    cublasStatus_t s = (call);                                               \
    if (s != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "%s:%d cublas error: %d\n", __FILE__, __LINE__,        \
              (int)s);                                                       \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define CUDNN_CHECK(call) do {                                               \
    cudnnStatus_t s = (call);                                                \
    if (s != CUDNN_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d cudnn error: %s\n", __FILE__, __LINE__,         \
              cudnnGetErrorString(s));                                       \
      abort();                                                               \
    }                                                                        \
  } while (0)

static int timing_enabled(void) {
  if (g_timing_enabled >= 0) return g_timing_enabled;
  const char *env = getenv("POLYGEIST_RT_TIMING");
  g_timing_enabled =
      env && env[0] != '\0' && strcmp(env, "0") != 0 &&
      strcmp(env, "false") != 0 && strcmp(env, "FALSE") != 0;
  return g_timing_enabled;
}

static FILE *timing_file(void) {
  if (!timing_enabled()) return NULL;
  if (g_timing_file) return g_timing_file;
  const char *path = getenv("POLYGEIST_RT_TIMING_FILE");
  if (path && path[0] != '\0') {
    g_timing_file = fopen(path, "a");
    if (!g_timing_file) {
      fprintf(stderr, "polygeist runtime: failed to open timing file %s\n", path);
      abort();
    }
  } else {
    g_timing_file = stderr;
  }
  return g_timing_file;
}

static double wall_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void timing_gpu_begin(void) {
  if (timing_enabled()) CUDA_CHECK(cudaEventRecord(g_ev_begin, g_stream));
}

static void timing_gpu_end(
    const char *op, int32_t m, int32_t n, int32_t k, double host_start_ms) {
  if (!timing_enabled()) {
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return;
  }

  CUDA_CHECK(cudaEventRecord(g_ev_end, g_stream));
  CUDA_CHECK(cudaEventSynchronize(g_ev_end));
  float device_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&device_ms, g_ev_begin, g_ev_end));

  FILE *f = timing_file();
  fprintf(f,
          "POLYGEIST_RT_TIMING\top=%s\tm=%d\tn=%d\tk=%d\t"
          "host_ms=%.6f\tdevice_ms=%.6f\n",
          op, (int)m, (int)n, (int)k, wall_time_ms() - host_start_ms,
          (double)device_ms);
  fflush(f);
}

static void timing_host_only(
    const char *op, int32_t m, int32_t n, int32_t k, double host_start_ms) {
  if (!timing_enabled()) return;
  FILE *f = timing_file();
  fprintf(f,
          "POLYGEIST_RT_TIMING\top=%s\tm=%d\tn=%d\tk=%d\t"
          "host_ms=%.6f\tdevice_ms=0.000000\n",
          op, (int)m, (int)n, (int)k, wall_time_ms() - host_start_ms);
  fflush(f);
}

static void ensure_cudnn(void) {
  if (g_cudnn) return;
  CUDNN_CHECK(cudnnCreate(&g_cudnn));
  CUDNN_CHECK(cudnnSetStream(g_cudnn, g_stream));
}

static void ensure_cublaslt(void) {
  if (g_lt) return;
  cublasStatus_t s = cublasLtCreate(&g_lt);
  if (s != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasLtCreate failed: %d\n", (int)s);
    abort();
  }
}

// Zero-copy helpers with PERSISTENT registration. cudaHostRegister has
// real cost on Jetson (page-table setup for the mapped range) — for an
// 8000×8000 double matrix that's 128K pages, ~50 ms per register call.
// Many kernels touch the same buffer multiple times (e.g. gemver:
// A is read/written by 2 gers + 2 gemvs = 4 shim calls). Re-registering
// + unregistering on every call is wasteful.
//
// Strategy: register on first use, NEVER unregister. The page mapping
// stays live for the rest of the program. Each shim call's first action
// is a fast no-op "already registered" check.
//
// Cache implementation: small open-addressed hash table keyed on host
// pointer. Size of 256 entries handles every benchmark we care about
// (polybench has ≤ 12 distinct buffers per kernel).

#define HOSTREG_CACHE_CAP 256
struct hostreg_entry { void *host; void *dev; size_t bytes; };
static struct hostreg_entry g_hostreg_cache[HOSTREG_CACHE_CAP];
static int g_hostreg_count = 0;

static int range_contains(void *outer, size_t outer_bytes,
                          void *inner, size_t inner_bytes) {
  uintptr_t o0 = (uintptr_t)outer;
  uintptr_t i0 = (uintptr_t)inner;
  uintptr_t o1 = o0 + outer_bytes;
  uintptr_t i1 = i0 + inner_bytes;
  return i0 >= o0 && i1 <= o1;
}

static int ranges_overlap(void *a, size_t a_bytes, void *b, size_t b_bytes) {
  uintptr_t a0 = (uintptr_t)a;
  uintptr_t b0 = (uintptr_t)b;
  uintptr_t a1 = a0 + a_bytes;
  uintptr_t b1 = b0 + b_bytes;
  return a0 < b1 && b0 < a1;
}

static void *hostreg_cache_lookup(void *ptr, size_t bytes) {
  for (int i = 0; i < g_hostreg_count; ++i) {
    struct hostreg_entry *e = &g_hostreg_cache[i];
    if (range_contains(e->host, e->bytes, ptr, bytes)) {
      uintptr_t delta = (uintptr_t)ptr - (uintptr_t)e->host;
      return (void *)((uintptr_t)e->dev + delta);
    }
  }
  return NULL;
}

static void hostreg_cache_remove_overlaps(void *ptr, size_t bytes) {
  for (int i = 0; i < g_hostreg_count;) {
    struct hostreg_entry *e = &g_hostreg_cache[i];
    if (!ranges_overlap(e->host, e->bytes, ptr, bytes)) {
      ++i;
      continue;
    }
    cudaError_t err = cudaHostUnregister(e->host);
    if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
      fprintf(stderr, "%s:%d cudaHostUnregister(%p) failed: %s\n",
              __FILE__, __LINE__, e->host, cudaGetErrorString(err));
      abort();
    }
    g_hostreg_cache[i] = g_hostreg_cache[g_hostreg_count - 1];
    g_hostreg_count--;
  }
}

static void hostreg_cache_insert(void *host, void *dev, size_t bytes) {
  if (g_hostreg_count >= HOSTREG_CACHE_CAP) {
    fprintf(stderr, "polygeist runtime: hostreg cache full (cap=%d)\n",
            HOSTREG_CACHE_CAP);
    abort();
  }
  g_hostreg_cache[g_hostreg_count].host = host;
  g_hostreg_cache[g_hostreg_count].dev  = dev;
  g_hostreg_cache[g_hostreg_count].bytes = bytes;
  g_hostreg_count++;
}

// We tried bypassing cudaHostRegister and passing host pointers directly
// to cuBLAS — fails with illegal-memory-access. cuBLAS requires the
// buffer to be registered (or device-allocated) even on a Tegra SoC
// where the iGPU can technically reach any DRAM page.
static void *register_host_safe(void *ptr, size_t bytes) {
  void *cached = hostreg_cache_lookup(ptr, bytes);
  if (cached) return cached;
  hostreg_cache_remove_overlaps(ptr, bytes);
  cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterMapped);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    fprintf(stderr, "%s:%d cudaHostRegister(%p, %zu) failed: %s\n",
            __FILE__, __LINE__, ptr, bytes, cudaGetErrorString(err));
    abort();
  }
  void *dev = NULL;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev, ptr, 0));
  hostreg_cache_insert(ptr, dev, bytes);
  return dev;
}

// Persistent-registration model: never unregister. Mappings live until
// the program exits, at which point the OS reclaims them anyway.
static void unregister_host_safe(void *ptr) { (void)ptr; }

static void destroy_backend_desc(cudnnBackendDescriptor_t *desc) {
  if (*desc) {
    cudnnBackendDestroyDescriptor(*desc);
    *desc = NULL;
  }
}

static void report_rmsnorm_backend_fallback(
    const char *where, cudnnStatus_t status) {
  static int warned = 0;
  if (warned) return;
  warned = 1;
  fprintf(stderr,
          "polygeist runtime: cuDNN RMSNorm graph unavailable at %s: %s; "
          "using host fallback\n",
          where, cudnnGetErrorString(status));
}

static int set_backend_attr(
    cudnnBackendDescriptor_t desc,
    cudnnBackendAttributeName_t attr,
    cudnnBackendAttributeType_t type,
    int64_t count,
    const void *value,
    const char *where,
    cudnnStatus_t *last_status) {
  cudnnStatus_t status =
      cudnnBackendSetAttribute(desc, attr, type, count, value);
  if (status != CUDNN_STATUS_SUCCESS) {
    *last_status = status;
    report_rmsnorm_backend_fallback(where, status);
    return 0;
  }
  return 1;
}

static int finalize_backend_desc(
    cudnnBackendDescriptor_t desc,
    const char *where,
    cudnnStatus_t *last_status) {
  cudnnStatus_t status = cudnnBackendFinalize(desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    *last_status = status;
    report_rmsnorm_backend_fallback(where, status);
    return 0;
  }
  return 1;
}

static int make_f32_backend_tensor(
    cudnnBackendDescriptor_t *desc,
    int64_t uid,
    const int64_t *dims,
    const int64_t *strides,
    int64_t rank,
    bool by_value,
    const char *name,
    cudnnStatus_t *last_status) {
  cudnnStatus_t status =
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, desc);
  if (status != CUDNN_STATUS_SUCCESS) {
    *last_status = status;
    report_rmsnorm_backend_fallback(name, status);
    return 0;
  }

  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  int64_t alignment = 4;
  if (!set_backend_attr(*desc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                        CUDNN_TYPE_DATA_TYPE, 1, &dtype, name,
                        last_status) ||
      !set_backend_attr(*desc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                        CUDNN_TYPE_INT64, rank, dims, name, last_status) ||
      !set_backend_attr(*desc, CUDNN_ATTR_TENSOR_STRIDES,
                        CUDNN_TYPE_INT64, rank, strides, name, last_status) ||
      !set_backend_attr(*desc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                        CUDNN_TYPE_INT64, 1, &uid, name, last_status) ||
      !set_backend_attr(*desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                        CUDNN_TYPE_INT64, 1, &alignment, name,
                        last_status))
    return 0;

  if (by_value &&
      !set_backend_attr(*desc, CUDNN_ATTR_TENSOR_IS_BY_VALUE,
                        CUDNN_TYPE_BOOLEAN, 1, &by_value, name,
                        last_status))
    return 0;

  return finalize_backend_desc(*desc, name, last_status);
}

void polygeist_cublas_init(void) {
  if (g_initialized) return;
  CUDA_CHECK(cudaStreamCreate(&g_stream));
  CUBLAS_CHECK(cublasCreate(&g_handle));
  CUBLAS_CHECK(cublasSetStream(g_handle, g_stream));
  CUBLAS_CHECK(cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_HOST));
  CUDA_CHECK(cudaEventCreate(&g_ev_begin));
  CUDA_CHECK(cudaEventCreate(&g_ev_end));
  g_initialized = 1;
}

void polygeist_cublas_destroy(void) {
  if (g_timing_file && g_timing_file != stderr) {
    fclose(g_timing_file);
    g_timing_file = NULL;
  }
  if (!g_initialized) return;
  cudaEventDestroy(g_ev_begin);
  cudaEventDestroy(g_ev_end);
  cublasDestroy(g_handle);
  cudaStreamDestroy(g_stream);
  g_initialized = 0;
}

void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_B = (size_t)K * (size_t)ldb * sizeof(double);
  size_t bytes_C = (size_t)M * (size_t)ldc * sizeof(double);

  // Pin host buffers for direct GPU access (zero-copy on Jetson).
  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dB = (double *)register_host_safe((void *)B, bytes_B);
  double *dC = (double *)register_host_safe(C, bytes_C);

  // Row-major C = α A·B + β C  →  col-major Cᵀ = α Bᵀ·Aᵀ + β Cᵀ
  timing_gpu_begin();
  CUBLAS_CHECK(cublasDgemm(g_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M, /*k=*/K,
                            &alpha,
                            dB, ldb,
                            dA, lda,
                            &beta,
                            dC, ldc));
  timing_gpu_end("cublasDgemm", M, N, K, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

void polygeist_cublas_sgemm(
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float *A, int32_t lda,
    const float *B, int32_t ldb,
    float beta,
    float *C, int32_t ldc) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(float);
  size_t bytes_B = (size_t)K * (size_t)ldb * sizeof(float);
  size_t bytes_C = (size_t)M * (size_t)ldc * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dB = (float *)register_host_safe((void *)B, bytes_B);
  float *dC = (float *)register_host_safe(C, bytes_C);

  timing_gpu_begin();
  CUBLAS_CHECK(cublasSgemm(g_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M, /*k=*/K,
                            &alpha,
                            dB, ldb,
                            dA, lda,
                            &beta,
                            dC, ldc));
  timing_gpu_end("cublasSgemm", M, N, K, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

// Host-side memset. In the current no-hoisting model the array lives on
// host between launches; pulling it to device just to zero is wasteful.
void polygeist_cublas_memset_zero_2d(int32_t M, int32_t N,
                                       double *A, int32_t lda) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  if (lda == N) {
    // Contiguous: one memset.
    memset(A, 0, (size_t)M * (size_t)N * sizeof(double));
  } else {
    for (int32_t i = 0; i < M; ++i) {
      memset(&A[(size_t)i * (size_t)lda], 0,
             (size_t)N * sizeof(double));
    }
  }
  timing_host_only("host_memset_zero_2d_f64", M, N, 0, host_start_ms);
}

// y = α*x + β*y (axpby). O(N) bandwidth-bound; H↔D copy + two cuBLAS
// calls would dominate any GPU benefit. Do it on the host directly.
void polygeist_cublas_daxpby(int32_t N, double alpha, const double *x,
                              double beta, double *y) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  for (int32_t i = 0; i < N; ++i) y[i] = alpha * x[i] + beta * y[i];
  timing_host_only("host_daxpby", N, 1, 0, host_start_ms);
}

// y += x (axpy with α=1).
void polygeist_cublas_daxpy_unit(int32_t N, const double *x, double *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)N * sizeof(double);
  double *dx = (double *)register_host_safe((void *)x, bytes);
  double *dy = (double *)register_host_safe(y, bytes);
  double one = 1.0;
  timing_gpu_begin();
  CUBLAS_CHECK(cublasDaxpy(g_handle, N, &one, dx, 1, dy, 1));
  timing_gpu_end("cublasDaxpy", N, 1, 0, host_start_ms);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

// Rank-2 update: A += u1·v1ᵀ + u2·v2ᵀ (gemver body). Two cublasDger calls.
void polygeist_cublas_dger_rank2(int32_t M, int32_t N,
                                   const double *u1, const double *v1,
                                   const double *u2, const double *v2,
                                   double *A, int32_t lda) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  double one = 1.0;
  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_u = (size_t)M * sizeof(double);
  size_t bytes_v = (size_t)N * sizeof(double);

  double *dA  = (double *)register_host_safe(A,         bytes_A);
  double *du1 = (double *)register_host_safe((void *)u1, bytes_u);
  double *dv1 = (double *)register_host_safe((void *)v1, bytes_v);
  double *du2 = (double *)register_host_safe((void *)u2, bytes_u);
  double *dv2 = (double *)register_host_safe((void *)v2, bytes_v);

  // Row-major A[i,j] += u1[i]*v1[j] + u2[i]*v2[j].
  // cuBLAS Dger col-major: pass (m=N, n=M, x=v, y=u) for row-major A += u·vᵀ.
  timing_gpu_begin();
  CUBLAS_CHECK(cublasDger(g_handle, /*m=*/N, /*n=*/M,
                          &one, dv1, 1, du1, 1, dA, lda));
  CUBLAS_CHECK(cublasDger(g_handle, /*m=*/N, /*n=*/M,
                          &one, dv2, 1, du2, 1, dA, lda));
  timing_gpu_end("cublasDger_rank2", M, N, 0, host_start_ms);

  unregister_host_safe(A);
  unregister_host_safe((void *)u1);
  unregister_host_safe((void *)v1);
  unregister_host_safe((void *)u2);
  unregister_host_safe((void *)v2);
}

// Host-side 1D memset. Same justification as the 2D variant — host copy
// to device just to zero is wasteful.
void polygeist_cublas_memset_zero_1d(int32_t N, double *v) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  memset(v, 0, (size_t)N * sizeof(double));
  timing_host_only("host_memset_zero_1d_f64", N, 1, 0, host_start_ms);
}

void polygeist_cublas_memset_zero_1d_f32(int32_t N, float *v) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  memset(v, 0, (size_t)N * sizeof(float));
  timing_host_only("host_memset_zero_1d_f32", N, 1, 0, host_start_ms);
}

// y = α·A·x + β·y, row-major.  Mirrors polygeist_cublas_dgemm structure
// (alloc → H2D → cuBLAS → D2H → free) but for the gemv shape.
//
// cuBLAS is column-major; row-major y = A·x is equivalent to a column-major
// `y = Aᵀ·x` view. Pass CUBLAS_OP_T with the row-major A's storage so cuBLAS
// reads it as the transposed column-major matrix — algebraically the same.
void polygeist_cublas_dgemv(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_x = (size_t)N * sizeof(double);
  size_t bytes_y = (size_t)M * sizeof(double);

  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dx = (double *)register_host_safe((void *)x, bytes_x);
  double *dy = (double *)register_host_safe(y, bytes_y);

  // Row-major y = A·x  →  col-major view of A is Aᵀ; OP_T undoes that.
  timing_gpu_begin();
  CUBLAS_CHECK(cublasDgemv(g_handle,
                            CUBLAS_OP_T,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  timing_gpu_end("cublasDgemv", M, N, 0, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

void polygeist_cublas_sgemv(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(float);
  size_t bytes_x = (size_t)N * sizeof(float);
  size_t bytes_y = (size_t)M * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dx = (float *)register_host_safe((void *)x, bytes_x);
  float *dy = (float *)register_host_safe(y, bytes_y);

  timing_gpu_begin();
  CUBLAS_CHECK(cublasSgemv(g_handle,
                            CUBLAS_OP_T,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  timing_gpu_end("cublasSgemv", M, N, 0, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

// y = α·Aᵀ·x + β·y, row-major. Shim signature is identical to the no-
// transpose dgemv shim; the only difference is the cuBLAS op flag.
//
// Row-major Aᵀ (logically N×M) · x (length M) → y (length N). The col-
// major view of row-major A IS Aᵀ, so we use CUBLAS_OP_N with the same
// (m=N, n=M, lda=lda_rowmajor) the no-transpose shim uses.
void polygeist_cublas_dgemv_T(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_x = (size_t)M * sizeof(double);   // x is M for Aᵀ·x
  size_t bytes_y = (size_t)N * sizeof(double);   // y is N for Aᵀ·x

  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dx = (double *)register_host_safe((void *)x, bytes_x);
  double *dy = (double *)register_host_safe(y, bytes_y);

  timing_gpu_begin();
  CUBLAS_CHECK(cublasDgemv(g_handle,
                            CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  timing_gpu_end("cublasDgemv_T", M, N, 0, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

void polygeist_cublas_sgemv_T(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(float);
  size_t bytes_x = (size_t)M * sizeof(float);
  size_t bytes_y = (size_t)N * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dx = (float *)register_host_safe((void *)x, bytes_x);
  float *dy = (float *)register_host_safe(y, bytes_y);

  timing_gpu_begin();
  CUBLAS_CHECK(cublasSgemv(g_handle,
                            CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  timing_gpu_end("cublasSgemv_T", M, N, 0, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

// Host-side scale. Could use cublasDscal but the H↔D copy overhead would
// dominate this O(MN) op; do it on the CPU side. Future device-residency
// hoisting will make this a GPU op.
void polygeist_cublas_dscal_2d(int32_t M, int32_t N, double scale,
                                 double *A, int32_t lda) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  for (int32_t i = 0; i < M; ++i) {
    double *row = &A[(size_t)i * (size_t)lda];
    for (int32_t j = 0; j < N; ++j) row[j] *= scale;
  }
  timing_host_only("host_dscal_2d", M, N, 0, host_start_ms);
}

// cuDNN 9-tap conv2d. Filter weights passed at runtime so the same shim
// handles polybench, Sobel, Gaussian, or any other 3x3 weighted conv.
// Single-image, single-channel, FP64, no-padding, stride-1.
void polygeist_cudnn_conv2d_3x3_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2,
    double w3, double w4, double w5,
    double w6, double w7, double w8,
    const double *A, double *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  // Caller-supplied filter (laid out row-major in the 3x3 grid).
  const double filter_h[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // 1 batch, 1 channel, M×N input; FP64 NCHW
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, M, N));
  // Filter: 1 out-ch, 1 in-ch, 3×3, FP64 NCHW
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_DOUBLE,
                                          CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
  // No padding, stride 1, dilation 1; use CROSS_CORRELATION (no flip)
  // since polybench's body matches cross-correlation semantics.
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, /*pad_h=*/0, /*pad_w=*/0, /*stride_h=*/1, /*stride_w=*/1,
      /*dilation_h=*/1, /*dilation_w=*/1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
  // Output: 1 batch, 1 channel, (M-2)×(N-2)
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, M - 2, N - 2));

  // Device allocations
  size_t bytes_in   = (size_t)M * (size_t)N * sizeof(double);
  size_t bytes_f    = 9 * sizeof(double);
  size_t bytes_out  = (size_t)(M - 2) * (size_t)(N - 2) * sizeof(double);
  double *dA = NULL, *dF = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dA, bytes_in));
  CUDA_CHECK(cudaMalloc((void**)&dF, bytes_f));
  CUDA_CHECK(cudaMalloc((void**)&dB, bytes_out));
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  // Algorithm choice: ask cuDNN for the best fwd algo it can serve.
  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      /*requestedAlgoCount=*/1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN: no fwd algo available for this shape\n");
    abort();
  }

  // Workspace
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  // Run
  double alpha = 1.0, beta = 0.0;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_9tap_f64", M, N, 9, host_start_ms);

  // The output (M-2)×(N-2) needs to be copied back into the *interior* of
  // B (i.e. B[1..M-2][1..N-2]) — that's what polybench's kernel writes to.
  // Copy row by row (N-2 doubles per row, into B + (i+1)*N + 1).
  for (int32_t i = 0; i < M - 2; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)(i + 1) * (size_t)N + 1,
        dB + (size_t)i * (size_t)(N - 2),
        (size_t)(N - 2) * sizeof(double),
        cudaMemcpyDeviceToHost, g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dA);  cudaFree(dF);  cudaFree(dB);
  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

// Backward-compat wrapper for the legacy hardcoded-weights call site.
// Forwards to the generic shim with polybench's filter.
void polygeist_cudnn_conv2d_polybench9tap(
    int32_t M, int32_t N, const double *A, double *B) {
  polygeist_cudnn_conv2d_3x3_f64(M, N,
     0.2,  0.5, -0.8,
    -0.3,  0.6, -0.9,
     0.4,  0.7,  0.1,
    A, B);
}

// FP32 variant — same structure as the f64 path, but with CUDNN_DATA_FLOAT
// descriptors and float*/cudaMemcpy for f32 buffers. On Ampere+ GPUs (Orin
// included) cuDNN uses tensor-core kernels for f32 conv, so this is the
// dtype to use for actual perf comparison (f64 falls back to a generic
// non-tensor-core path).
void polygeist_cudnn_conv2d_3x3_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2,
    float w3, float w4, float w5,
    float w6, float w7, float w8,
    const float *A, float *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  const float filter_h[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, M - 2, N - 2));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(float);
  size_t bytes_f   = 9 * sizeof(float);
  size_t bytes_out = (size_t)(M - 2) * (size_t)(N - 2) * sizeof(float);
  float *dA = NULL, *dF = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dA, bytes_in));
  CUDA_CHECK(cudaMalloc((void**)&dF, bytes_f));
  CUDA_CHECK(cudaMalloc((void**)&dB, bytes_out));
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f32): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_9tap_f32", M, N, 9, host_start_ms);

  for (int32_t i = 0; i < M - 2; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)(i + 1) * (size_t)N + 1,
        dB + (size_t)i * (size_t)(N - 2),
        (size_t)(N - 2) * sizeof(float),
        cudaMemcpyDeviceToHost, g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dA);  cudaFree(dF);  cudaFree(dB);
  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

// FP16 variant. cuDNN tensor cores light up here on Ampere+ (Orin) when the
// shape is large enough and channel-aligned. Single-batch single-channel may
// still fall back to a generic path — but for batched/channeled workloads
// this is the fast path. Math/accumulation type is FP32 inside cuDNN.
// Guarded on __FLT16_MAX__ to match the header declaration.
#if defined(__FLT16_MAX__)
void polygeist_cudnn_conv2d_3x3_f16(
    int32_t M, int32_t N,
    _Float16 w0, _Float16 w1, _Float16 w2,
    _Float16 w3, _Float16 w4, _Float16 w5,
    _Float16 w6, _Float16 w7, _Float16 w8,
    const _Float16 *A, _Float16 *B) {
  polygeist_cublas_init();
  ensure_cudnn();

  // Reinterpret host-side _Float16 → uint16_t (identical bit layout). cuDNN
  // reads the buffer as CUDNN_DATA_HALF via the descriptor, so the type of
  // the device pointer doesn't matter as long as the bits are right.
  uint16_t filter_h[9];
  __builtin_memcpy(&filter_h[0], &w0, 2);
  __builtin_memcpy(&filter_h[1], &w1, 2);
  __builtin_memcpy(&filter_h[2], &w2, 2);
  __builtin_memcpy(&filter_h[3], &w3, 2);
  __builtin_memcpy(&filter_h[4], &w4, 2);
  __builtin_memcpy(&filter_h[5], &w5, 2);
  __builtin_memcpy(&filter_h[6], &w6, 2);
  __builtin_memcpy(&filter_h[7], &w7, 2);
  __builtin_memcpy(&filter_h[8], &w8, 2);

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_HALF,
                                          CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
  // Accumulate in FP32 inside the conv (CUDNN_DATA_FLOAT compute dtype).
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_HALF, 1, 1, M - 2, N - 2));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(uint16_t);
  size_t bytes_f   = 9 * sizeof(uint16_t);
  size_t bytes_out = (size_t)(M - 2) * (size_t)(N - 2) * sizeof(uint16_t);
  uint16_t *dA = NULL, *dF = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dA, bytes_in));
  CUDA_CHECK(cudaMalloc((void**)&dF, bytes_f));
  CUDA_CHECK(cudaMalloc((void**)&dB, bytes_out));
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f16): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  // cuDNN expects FP32 alpha/beta scalars when the compute dtype is FP32,
  // regardless of the I/O dtype.
  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));

  for (int32_t i = 0; i < M - 2; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        (void*)((uint16_t*)B + (size_t)(i + 1) * (size_t)N + 1),
        dB + (size_t)i * (size_t)(N - 2),
        (size_t)(N - 2) * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dA);  cudaFree(dF);  cudaFree(dB);
  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}
#endif  // __FLT16_MAX__

#if defined(__BFLT16_MAX__) || defined(__ARM_FEATURE_BF16) || \
    defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || defined(__BF16__)
// BF16 variant. Same structure as the FP16 path but with CUDNN_DATA_BFLOAT16
// for I/O and filter. Compute dtype is still FP32 (BF16 has the same exponent
// range as FP32, so the FP32 accumulator avoids overflow without needing
// rescaling).
void polygeist_cudnn_conv2d_3x3_bf16(
    int32_t M, int32_t N,
    __bf16 w0, __bf16 w1, __bf16 w2,
    __bf16 w3, __bf16 w4, __bf16 w5,
    __bf16 w6, __bf16 w7, __bf16 w8,
    const __bf16 *A, __bf16 *B) {
  polygeist_cublas_init();
  ensure_cudnn();

  // Host-side __bf16 → uint16_t bit-copy. Same trick as the f16 path; cuDNN
  // reads CUDNN_DATA_BFLOAT16 via the descriptor, the underlying buffer
  // type doesn't matter on the C side.
  uint16_t filter_h[9];
  __builtin_memcpy(&filter_h[0], &w0, 2);
  __builtin_memcpy(&filter_h[1], &w1, 2);
  __builtin_memcpy(&filter_h[2], &w2, 2);
  __builtin_memcpy(&filter_h[3], &w3, 2);
  __builtin_memcpy(&filter_h[4], &w4, 2);
  __builtin_memcpy(&filter_h[5], &w5, 2);
  __builtin_memcpy(&filter_h[6], &w6, 2);
  __builtin_memcpy(&filter_h[7], &w7, 2);
  __builtin_memcpy(&filter_h[8], &w8, 2);

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_BFLOAT16, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_BFLOAT16,
                                          CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_BFLOAT16, 1, 1, M - 2, N - 2));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(uint16_t);
  size_t bytes_f   = 9 * sizeof(uint16_t);
  size_t bytes_out = (size_t)(M - 2) * (size_t)(N - 2) * sizeof(uint16_t);
  uint16_t *dA = NULL, *dF = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dA, bytes_in));
  CUDA_CHECK(cudaMalloc((void**)&dF, bytes_f));
  CUDA_CHECK(cudaMalloc((void**)&dB, bytes_out));
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(bf16): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));

  for (int32_t i = 0; i < M - 2; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        (void*)((uint16_t*)B + (size_t)(i + 1) * (size_t)N + 1),
        dB + (size_t)i * (size_t)(N - 2),
        (size_t)(N - 2) * sizeof(uint16_t),
        cudaMemcpyDeviceToHost, g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dA);  cudaFree(dF);  cudaFree(dB);
  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}
#endif  // bf16 support

// INT32 variant.
//
// IMPORTANT: cuDNN's `cudnnConvolutionForward` does NOT support a pure
// INT32 input + INT32 filter + INT32 compute configuration. On Orin
// (Ampere) the call to `cudnnSetTensor4dDescriptor(..., CUDNN_DATA_INT32,
// ...)` (or, equivalently, the convolution-descriptor setup with
// CUDNN_DATA_INT32 as the compute type) returns CUDNN_STATUS_BAD_PARAM —
// not because of any error in our argument values, but because cuDNN
// simply doesn't expose INT32 as a standalone fwd-conv I/O dtype.
//
// Where INT32 *does* appear in cuDNN's API is as the *accumulator* dtype
// for an INT8 input × INT8 filter via `cudnnConvolutionBiasActivationForward`
// (and NHWC_VECT_C layouts). That's a fundamentally different API surface
// — different operand layout, requires quantising the user's int input
// down to INT8 with a scale factor, etc. — so we don't silently rewrite
// the user's INT32 stencil into INT8 quant.
//
// Consequently this function intentionally fails fast at the cuDNN call:
// no host-side fallback, no silent reroute. The matcher/rewriter/ABI
// lowering pipeline still exercises end-to-end — verifiable by inspecting
// the produced `func.call @polygeist_cudnn_conv2d_3x3_i32` op — but the
// GPU side is "not implemented" until a real INT32 conv path lands.
// Options for that follow-up:
//   * Hand-written CUDA kernel (small .cu compiled with nvcc; the runtime
//     loads it via cuModuleLoad + cuLaunchKernel).
//   * Switch to cuDNN INT8 quant path (changes the user-visible dtype).
//   * Use a different library (cutlass, raw CUB) that supports INT32 conv.
void polygeist_cudnn_conv2d_3x3_i32(
    int32_t M, int32_t N,
    int32_t w0, int32_t w1, int32_t w2,
    int32_t w3, int32_t w4, int32_t w5,
    int32_t w6, int32_t w7, int32_t w8,
    const int32_t *A, int32_t *B) {
  polygeist_cublas_init();
  ensure_cudnn();

  const int32_t filter_h[9] = { w0, w1, w2, w3, w4, w5, w6, w7, w8 };
  (void)A; (void)B; (void)filter_h;  // silence unused until cuDNN call below.

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // This is the call that will trip CUDNN_STATUS_BAD_PARAM on Orin/Ampere
  // for the pure-INT32 configuration. We deliberately do not catch the
  // error — the CUDNN_CHECK macro will print the cuDNN message and abort,
  // making the unsupported-dtype failure visible to the caller.
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_INT32, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_INT32,
                                          CUDNN_TENSOR_NCHW, 1, 1, 3, 3));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
  // If by some firmware/cuDNN-version combination the above three calls
  // succeed, we'd still need to run the actual conv. The pre-existing
  // code path for the float dtypes (algo selection, workspace alloc,
  // cudnnConvolutionForward, async memcpy back) would go here. Until
  // INT32 is supported we leave this as a hard failure — `CUDNN_CHECK`
  // above will have aborted before reaching this point.
  fprintf(stderr,
          "polygeist_cudnn_conv2d_3x3_i32: cuDNN unexpectedly accepted "
          "INT32 descriptors but the conv body is not implemented.\n");
  abort();
}

// INT16 variant. cuDNN has no INT16 conv path. We upcast inputs/filter to
// INT32 on the host, then delegate to `polygeist_cudnn_conv2d_3x3_i32`.
// That i32 shim is itself NOT implemented on the GPU (see the long
// comment above it — cuDNN doesn't expose INT32 forward conv either), so
// the i16 path also fails at the same cuDNN call. The upcast is still
// the right structure once a real INT32 GPU kernel lands; only the
// underlying i32 path needs replacing.
void polygeist_cudnn_conv2d_3x3_i16(
    int32_t M, int32_t N,
    int16_t w0, int16_t w1, int16_t w2,
    int16_t w3, int16_t w4, int16_t w5,
    int16_t w6, int16_t w7, int16_t w8,
    const int16_t *A, int16_t *B) {
  // Upcast input to i32.
  size_t total = (size_t)M * (size_t)N;
  int32_t *A32 = (int32_t*)malloc(total * sizeof(int32_t));
  int32_t *B32 = (int32_t*)malloc(total * sizeof(int32_t));
  if (!A32 || !B32) { fprintf(stderr, "i16 shim: oom\n"); abort(); }
  for (size_t k = 0; k < total; ++k) A32[k] = (int32_t)A[k];
  // Zero B32's interior so the cuDNN write hits a known starting state;
  // the borders won't be touched by the conv, and we won't copy them back.
  memset(B32, 0, total * sizeof(int32_t));

  polygeist_cudnn_conv2d_3x3_i32(M, N,
      (int32_t)w0, (int32_t)w1, (int32_t)w2,
      (int32_t)w3, (int32_t)w4, (int32_t)w5,
      (int32_t)w6, (int32_t)w7, (int32_t)w8,
      A32, B32);

  // Downcast i32 result back to i16 (interior only — borders are caller-owned).
  for (int32_t i = 1; i < M - 1; ++i) {
    for (int32_t j = 1; j < N - 1; ++j) {
      size_t k = (size_t)i * (size_t)N + (size_t)j;
      B[k] = (int16_t)B32[k];
    }
  }
  free(A32);
  free(B32);
}

// ============================================================================
// Extracted-darknet batched CNN-block primitives. All FP32, NCHW.
//
// MEMORY MODEL: same zero-copy pattern as the BLAS shims —
// cudaHostRegister + cudaHostGetDevicePointer via register_host_safe().
// On Jetson Orin's iGPU these calls just set up the page-table mapping
// (no bytes move). For workspace + descriptor allocations we use
// cudaMalloc/cudaFree (per-call); a future device-residency hoisting
// pass would amortize these across consecutive layers.
// ============================================================================

void polygeist_cudnn_conv2d_batched(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F, float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;

  size_t bytes_A   = (size_t)B  * IC * H  * W  * sizeof(float);
  size_t bytes_F   = (size_t)OC * IC * K  * K  * sizeof(float);
  size_t bytes_Out = (size_t)B  * OC * OH * OW * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A,  bytes_A);
  float *dF = (float *)register_host_safe((void *)F,  bytes_F);
  float *dO = (float *)register_host_safe(Out,        bytes_Out);

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, IC, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, OC, IC, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, OC, OH, OW));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN conv2d_batched: no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dO));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv2d_im2col_gemm_f32(
    int32_t IC, int32_t H, int32_t W, int32_t OC,
    int32_t K, int32_t S, int32_t P,
    const float *A, const float *F, float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  const int32_t OH = (H + 2 * P - K) / S + 1;
  const int32_t OW = (W + 2 * P - K) / S + 1;
  size_t bytes_A   = (size_t)IC * H * W * sizeof(float);
  size_t bytes_F   = (size_t)OC * IC * K * K * sizeof(float);
  size_t bytes_Out = (size_t)OC * OH * OW * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dF = (float *)register_host_safe((void *)F, bytes_F);
  float *dO = (float *)register_host_safe(Out, bytes_Out);

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, IC, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, OC, IC, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, P, P, S, S, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, OC, OH, OW));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN conv2d_im2col_gemm: no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dO));
  timing_gpu_end("cudnnConv2d_im2col_gemm", OC, OH * OW, IC * K * K,
                 host_start_ms);

  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_maxpool_batched(
    int32_t B, int32_t C, int32_t H, int32_t W, int32_t OH, int32_t OW,
    const float *A, float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  // Derive S = H / OH (common K==S case for our extracted kernels).
  int32_t S = H / OH;
  int32_t K = (S > 0) ? S : 2;

  size_t bytes_A   = (size_t)B * C * H  * W  * sizeof(float);
  size_t bytes_Out = (size_t)B * C * OH * OW * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dO = (float *)register_host_safe(Out,       bytes_Out);

  cudnnTensorDescriptor_t  in_desc, out_desc;
  cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, H, W));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, OH, OW));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(
      pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
      K, K, 0, 0, S, S));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnPoolingForward(
      g_cudnn, pool_desc, &alpha, in_desc, dA,
      &beta, out_desc, dO));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pool_desc);
}

void polygeist_cudnn_batchnorm_inference(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  // cuDNN expects (mean, variance) and an epsilon, computing
  //   y = scale * (x - mean) / sqrt(var + eps) + bias.
  // Our kernel was given (mean, inv_std) where inv_std = 1/sqrt(var+eps).
  // We invert: var = 1/inv_std² - eps. Use the same eps the caller used.
  // The standard ResNet/PyTorch eps is 1e-5.
  const double eps = 1e-5;

  float *var_h = (float *)malloc((size_t)C * sizeof(float));
  for (int32_t c = 0; c < C; ++c) {
    double s = (double)inv_std[c];
    double v = 1.0 / (s * s) - eps;
    if (v < 0) v = 0;
    var_h[c] = (float)v;
  }

  size_t bytes_x = (size_t)B * C * H * W * sizeof(float);
  size_t bytes_c = (size_t)C * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A,     bytes_x);
  float *dS = (float *)register_host_safe((void *)scale, bytes_c);
  float *dM = (float *)register_host_safe((void *)mean,  bytes_c);
  float *dB = (float *)register_host_safe((void *)bias,  bytes_c);
  float *dO = (float *)register_host_safe(Out,           bytes_x);
  float *dV = NULL;
  CUDA_CHECK(cudaMalloc((void **)&dV, bytes_c));
  CUDA_CHECK(cudaMemcpyAsync(dV, var_h, bytes_c,
                             cudaMemcpyHostToDevice, g_stream));

  cudnnTensorDescriptor_t x_desc, y_desc, bn_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, H, W));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, H, W));
  // bnScaleBiasMeanVarDesc: 1×C×1×1
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, C, 1, 1));

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      g_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
      x_desc, dA, y_desc, dO, bn_desc, dS, dB, dM, dV, eps));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dV);
  free(var_h);
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
  cudnnDestroyTensorDescriptor(bn_desc);
}

void polygeist_cudnn_add_tensor_batched(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A, float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  size_t bytes = (size_t)B * C * H * W * sizeof(float);
  float *dA = (float *)register_host_safe((void *)A, bytes);
  float *dO = (float *)register_host_safe(Out,       bytes);

  cudnnTensorDescriptor_t a_desc, o_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&a_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&o_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(a_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, H, W));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(o_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, C, H, W));

  // cudnnAddTensor computes Out = α*A + β*Out. We want Out += A, so α=β=1.
  float alpha = 1.0f, beta = 1.0f;
  CUDNN_CHECK(cudnnAddTensor(g_cudnn, &alpha, a_desc, dA,
                              &beta, o_desc, dO));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudnnDestroyTensorDescriptor(a_desc);
  cudnnDestroyTensorDescriptor(o_desc);
}

// Fused conv + bias + residual-add + relu via the SAME cuDNN API.
// y = activation(α₁·conv(x,w) + α₂·z + bias). We just feed real bias +
// real Z; no BN-folding step needed.
void polygeist_cudnn_conv_bias_relu_add_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *bias, const float *Z,
    float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;

  size_t bytes_A  = (size_t)B  * IC * H  * W  * sizeof(float);
  size_t bytes_F  = (size_t)OC * IC * K  * K  * sizeof(float);
  size_t bytes_Ou = (size_t)B  * OC * OH * OW * sizeof(float);
  size_t bytes_b  = (size_t)OC * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A,    bytes_A);
  float *dF = (float *)register_host_safe((void *)F,    bytes_F);
  float *dB = (float *)register_host_safe((void *)bias, bytes_b);
  float *dZ = (float *)register_host_safe((void *)Z,    bytes_Ou);
  float *dO = (float *)register_host_safe(Out,          bytes_Ou);

  cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnActivationDescriptor_t  act_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, IC, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, OC, IC, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, OC, OH, OW));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, OC, 1, 1));
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  // Algo selection — see the stack-smash note in
  // polygeist_cudnn_conv_bn_relu_fused for why this loop allocates an
  // array of ALGO_CANDIDATES not a single struct.
  enum { ALGO_CANDIDATES = 8 };
  cudnnConvolutionFwdAlgoPerf_t algos[ALGO_CANDIDATES];
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      ALGO_CANDIDATES, &n_returned, algos));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN conv_bias_relu_add: no fwd algo\n"); abort();
  }
  cudnnConvolutionFwdAlgo_t algo = algos[0].algo;
  for (int i = 0; i < n_returned; ++i)
    if (algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
      algo = algos[i].algo; break;
    }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  // y = relu(1·conv(A, F) + 1·Z + bias).
  float alpha1 = 1.0f, alpha2 = 1.0f;
  CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
      g_cudnn, &alpha1, in_desc, dA, f_desc, dF, conv_desc, algo,
      dWS, ws_size, &alpha2, out_desc, dZ,
      bias_desc, dB, act_desc, out_desc, dO));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  if (dWS) cudaFree(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyActivationDescriptor(act_desc);
}

void polygeist_cublas_memset_zero_2d_f32(int32_t M, int32_t N, float *A, int32_t lda) {
  /* Host memset — same as the f64 path. */
  if (lda == N) {
    memset(A, 0, (size_t)M * (size_t)N * sizeof(float));
  } else {
    for (int32_t i = 0; i < M; ++i)
      memset(&A[(size_t)i * (size_t)lda], 0, (size_t)N * sizeof(float));
  }
}

// 1×1 conv routed to batched gemm. For NCHW input (B, IC, H, W) and
// filter (OC, IC, 1, 1), each batch slice is a regular
// (OC, HW) = (OC, IC) × (IC, HW) gemm. F is shared across batches
// (stride 0); A and C each stride by their per-batch element count.
//
// Row-major / col-major swap, same trick as cublasDgemm: the col-major
// view of our row-major A_b (IC × HW) is (HW × IC), of F (OC × IC) is
// (IC × OC), of C_b (OC × HW) is (HW × OC). So:
//   col-major C_b (HW, OC) = α · col-major A_b (HW, IC) · F (IC, OC)
// → cublasSgemmStridedBatched(OP_N, OP_N, m=HW, n=OC, k=IC,
//                             α, A, lda=HW, A_stride=IC*HW,
//                             F, ldb=IC, F_stride=0,
//                             β, C, ldc=HW, C_stride=OC*HW,
//                             batchCount=B)
void polygeist_cublas_sgemm_1x1conv(
    int32_t B, int32_t IC, int32_t OC, int32_t HW,
    const float *A, const float *F, float *C) {
  polygeist_cublas_init();

  size_t bytes_A = (size_t)B  * IC * HW * sizeof(float);
  size_t bytes_F = (size_t)OC * IC * sizeof(float);
  size_t bytes_C = (size_t)B  * OC * HW * sizeof(float);
  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dF = (float *)register_host_safe((void *)F, bytes_F);
  float *dC = (float *)register_host_safe(C,         bytes_C);

  float alpha = 1.0f, beta = 0.0f;
  long long strideA = (long long)IC * HW;
  long long strideF = 0;
  long long strideC = (long long)OC * HW;
  CUBLAS_CHECK(cublasSgemmStridedBatched(g_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      HW, OC, IC,
      &alpha, dA, HW, strideA,
              dF, IC, strideF,
      &beta,  dC, HW, strideC,
      B));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

// AᵀA → cublasSsyrk_v2 (FP32). Half the flops of the equivalent
// gemm because syrk only computes the upper triangle of the symmetric
// output. cublasSsyrk's signature:
//   C = α·op(A)·op(A)ᵀ + β·C
// where uplo selects which triangle is touched.
//
// Row-major → col-major: our A is row-major (K×N), so its column-major
// view is Aᵀ (N×K). To compute row-major C[N,N] = Aᵀ·A we ask cublas
// to compute col-major Cᵀ[N,N] = (Aᵀ_col_view)·(A_col_view) = A_row·Aᵀ_row.
// Equivalent: pass A with op=N, treat as col-major (N rows × K cols).
// uplo = LOWER on the col-major matrix == UPPER on the row-major view.
// We fill in the missing triangle on host after the call so the caller
// sees a fully-populated symmetric matrix.
void polygeist_cublas_dsyrk(int32_t N, int32_t K, const float *A, float *C) {
  polygeist_cublas_init();

  size_t bytes_A = (size_t)K * N * sizeof(float);
  size_t bytes_C = (size_t)N * N * sizeof(float);
  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dC = (float *)register_host_safe(C,         bytes_C);

  float alpha = 1.0f, beta = 0.0f;
  // Layout math:
  //   Our C is row-major. cublas operates col-major. The SAME bytes
  //   look transposed: row-major C[i,j] is at byte i + j*N in col-major.
  //   cublasSsyrk(uplo=UPPER) writes col-major UPPER (i ≤ j) which maps
  //   to row-major positions (j, i) with j ≥ i — i.e. row-major LOWER.
  //   The mirror loop below then copies row-major lower → row-major upper.
  CUBLAS_CHECK(cublasSsyrk(g_handle,
                            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                            N, K,
                            &alpha, dA, N,
                            &beta,  dC, N));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  for (int32_t i = 0; i < N; ++i)
    for (int32_t j = i + 1; j < N; ++j)
      C[(size_t)i * N + j] = C[(size_t)j * N + i];
}

// Fused matmul + bias + relu via cublasLtMatmul with EPILOGUE_RELU_BIAS.
//
// Row-major to col-major: we compute Cᵀ = Bᵀ·Aᵀ + bias' the same way
// cublasDgemm does in this codebase — by swapping A↔B and treating
// "rows" of cublasLt's matrix as columns of ours. cublasLt's matmul
// descriptor uses col-major by default, so:
//   our row-major C[M,N] = A[M,K] · B[K,N]
//   ≡ col-major Cᵀ[N,M] = Bᵀ[N,K] · Aᵀ[K,M]
// With both A and B passed as CUBLAS_OP_N (no transpose flag), and the
// matrix layouts created in col-major with swapped sizes, the math
// works out exactly. bias[N] is a single per-output-column vector;
// cublasLt's RELU_BIAS epilogue applies it per column of the output.
void polygeist_cublaslt_matmul_bias_relu(
    int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, const float *bias,
    float *C) {
  polygeist_cublas_init();
  ensure_cublaslt();

  size_t bytes_A = (size_t)M * K * sizeof(float);
  size_t bytes_B = (size_t)K * N * sizeof(float);
  size_t bytes_C = (size_t)M * N * sizeof(float);
  size_t bytes_b = (size_t)N * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A,    bytes_A);
  float *dB = (float *)register_host_safe((void *)B,    bytes_B);
  float *dC = (float *)register_host_safe(C,            bytes_C);
  float *dBias = (float *)register_host_safe((void *)bias, bytes_b);

  cublasLtMatmulDesc_t   matmul_desc = NULL;
  cublasLtMatrixLayout_t aDesc = NULL, bDesc = NULL, cDesc = NULL;

  // Op descriptor: f32 compute, f32 scale.
  cublasStatus_t s;
  s = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasLtMatmulDescCreate failed: %d\n", (int)s); abort(); }

  cublasOperation_t opN = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                  &opN, sizeof(opN));
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                  &opN, sizeof(opN));

  // Epilogue: bias + ReLU (applied in that order, then ReLU on top of bias).
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                  &epi, sizeof(epi));
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                  &dBias, sizeof(dBias));

  // Row-major → col-major operand swap (same as cublasDgemm in this file):
  // Compute Cᵀ = Bᵀ_col · Aᵀ_col, where each is created as col-major with
  // sizes that mirror our row-major source. So in cublasLt's view:
  //   "A" of the matmul is our B (size N × K, col-major, lda=N=ldb_row)
  //   "B" of the matmul is our A (size K × M, col-major, lda=K)
  //   "C" of the matmul is our C (size N × M, col-major, lda=N)
  cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, N, K, N);
  cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_32F, K, M, K);
  cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, N, M, N);

  // Algorithm selection — heuristic, request 1 candidate.
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws_size = 16 * 1024 * 1024;  // 16 MB workspace
  cublasLtMatmulPreferenceSetAttribute(pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));
  cublasLtMatmulHeuristicResult_t heur;
  int n_results = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, matmul_desc,
      aDesc, bDesc, cDesc, cDesc, pref, 1, &heur, &n_results);
  if (n_results < 1) {
    fprintf(stderr, "cublasLt: no matmul algo available\n"); abort();
  }
  void *dWS = NULL;
  if (heur.workspaceSize > 0) CUDA_CHECK(cudaMalloc(&dWS, heur.workspaceSize));

  float alpha = 1.0f, beta = 0.0f;
  s = cublasLtMatmul(g_lt, matmul_desc,
      &alpha, dB, aDesc,    // swapped: cublasLt's "A" is our B
              dA, bDesc,    // swapped: cublasLt's "B" is our A
      &beta,  dC, cDesc,
              dC, cDesc,
      &heur.algo, dWS, heur.workspaceSize, g_stream);
  if (s != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasLtMatmul failed: %d\n", (int)s); abort();
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  if (dWS) cudaFree(dWS);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(aDesc);
  cublasLtMatrixLayoutDestroy(bDesc);
  cublasLtMatrixLayoutDestroy(cDesc);
  cublasLtMatmulDescDestroy(matmul_desc);
}

// Fused conv + bn-inference + relu via cudnnConvolutionBiasActivationForward.
// The trick is "BN folding": cudnnConvolutionBiasActivationForward computes
//   y = activation(α₁ * conv(x, w) + α₂ * z + bias)
// natively. To fold inference-mode BN into it, pre-compute on host:
//   w'[oc,ic,kh,kw] = w[oc,ic,kh,kw] * scale[oc] * inv_std[oc]
//   b'[oc]         = bias[oc] - scale[oc] * mean[oc] * inv_std[oc]
// Then cudnnConvolutionBiasActivationForward(x, w', 1, conv, 0, _, b',
// RELU, y) computes exactly relu(scale*(conv(x,w) - mean)*inv_std + bias).
//
// The folding is O(OC*IC*K²) on host, much smaller than the conv itself
// (the LARGE shape has IC=OC=64, K=3 → 36864 muls; the conv itself does
// ~10B muls). So it doesn't bottleneck. In a real CNN, this folding
// would be done once at model-load time, not per call.
void polygeist_cudnn_conv_bn_relu_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out) {
  polygeist_cublas_init();
  ensure_cudnn();

  const int32_t OH = H - K + 1;
  const int32_t OW = W - K + 1;

  // Host-side BN-into-conv folding.
  size_t n_w = (size_t)OC * IC * K * K;
  float *F_fold = (float *)malloc(n_w * sizeof(float));
  float *b_fold = (float *)malloc((size_t)OC * sizeof(float));
  for (int32_t oc = 0; oc < OC; ++oc) {
    float coef = scale[oc] * inv_std[oc];
    for (int32_t ic = 0; ic < IC; ++ic)
      for (int32_t kh = 0; kh < K; ++kh)
        for (int32_t kw = 0; kw < K; ++kw) {
          size_t idx = ((size_t)oc * IC + ic) * K * K +
                       (size_t)kh * K + kw;
          F_fold[idx] = F[idx] * coef;
        }
    b_fold[oc] = bias[oc] - scale[oc] * mean[oc] * inv_std[oc];
  }

  size_t bytes_A  = (size_t)B  * IC * H  * W  * sizeof(float);
  size_t bytes_F  = (size_t)OC * IC * K  * K  * sizeof(float);
  size_t bytes_Ou = (size_t)B  * OC * OH * OW * sizeof(float);
  size_t bytes_b  = (size_t)OC * sizeof(float);

  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dO = (float *)register_host_safe(Out,       bytes_Ou);
  // Folded weights / bias live on the device (recomputed per call —
  // could be hoisted to a one-time setup once we wire device-residency).
  float *dF = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc((void **)&dF, bytes_F));
  CUDA_CHECK(cudaMalloc((void **)&dB, bytes_b));
  CUDA_CHECK(cudaMemcpyAsync(dF, F_fold, bytes_F, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, b_fold, bytes_b, cudaMemcpyHostToDevice, g_stream));

  cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnActivationDescriptor_t  act_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, IC, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, OC, IC, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  // CUDNN_DEFAULT_MATH would let cuDNN pick tensor cores. Required for
  // the fused path on Ampere+ (Orin); without it the API falls back to
  // generic kernels.
  CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, B, OC, OH, OW));
  // Bias is 1×OC×1×1 broadcast across (B, OH, OW).
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, OC, 1, 1));
  // ReLU activation, no NaN propagation, threshold 0.
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

  // Algorithm selection. cudnnConvolutionBiasActivationForward requires
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM in many cuDNN versions
  // (the other algos return NOT_SUPPORTED through the fused API). Ask
  // cuDNN for up to 8 candidates in one call and pick PRECOMP_GEMM if
  // it appears; else fall back to cuDNN's first preference.
  enum { ALGO_CANDIDATES = 8 };
  cudnnConvolutionFwdAlgoPerf_t algos[ALGO_CANDIDATES];
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc,
      ALGO_CANDIDATES, &n_returned, algos));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN conv_bn_relu_fused: no fwd algo available\n");
    abort();
  }
  cudnnConvolutionFwdAlgo_t algo = algos[0].algo;
  for (int i = 0; i < n_returned; ++i) {
    if (algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
      algo = algos[i].algo;
      break;
    }
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) CUDA_CHECK(cudaMalloc(&dWS, ws_size));

  // y = act(α₁ * conv(x, w') + α₂ * z + b'). We want α₂ = 0 so z is
  // unused — but cuDNN requires a valid z descriptor + pointer anyway.
  // Reuse the output buffer as z (cuDNN accepts that when α₂ = 0).
  float alpha1 = 1.0f, alpha2 = 0.0f;
  CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
      g_cudnn, &alpha1, in_desc, dA, f_desc, dF, conv_desc, algo,
      dWS, ws_size, &alpha2, out_desc, dO,
      bias_desc, dB, act_desc, out_desc, dO));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  if (dWS) cudaFree(dWS);
  cudaFree(dF);
  cudaFree(dB);
  free(F_fold);
  free(b_fold);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyActivationDescriptor(act_desc);
}

static void rmsnorm_host_f32(
    int32_t N, const float *X, const float *Weight, float *Out) {
  float ss = 0.0f;
  for (int32_t i = 0; i < N; ++i)
    ss += X[i] * X[i];
  float scale = 1.0f / sqrtf(ss / (float)N + 1.0e-5f);
  for (int32_t i = 0; i < N; ++i)
    Out[i] = Weight[i] * (scale * X[i]);
}

#define RMSNORM_F32_CACHE_CAP 8
struct rmsnorm_f32_plan {
  int in_use;
  int unsupported;
  int32_t N;
  size_t bytes;
  float epsilon;

  float *dX;
  float *dWeight;
  float *dOut;
  float *dBias;
  void *workspace;

  cudnnBackendDescriptor_t x_desc;
  cudnnBackendDescriptor_t scale_desc;
  cudnnBackendDescriptor_t bias_desc;
  cudnnBackendDescriptor_t epsilon_desc;
  cudnnBackendDescriptor_t y_desc;
  cudnnBackendDescriptor_t norm_op;
  cudnnBackendDescriptor_t op_graph;
  cudnnBackendDescriptor_t engine;
  cudnnBackendDescriptor_t engine_cfg;
  cudnnBackendDescriptor_t plan;
  cudnnBackendDescriptor_t variant_pack;
};

static struct rmsnorm_f32_plan g_rmsnorm_f32_cache[RMSNORM_F32_CACHE_CAP];

static void release_rmsnorm_f32_plan_resources(struct rmsnorm_f32_plan *p) {
  destroy_backend_desc(&p->variant_pack);
  destroy_backend_desc(&p->plan);
  destroy_backend_desc(&p->engine_cfg);
  destroy_backend_desc(&p->engine);
  destroy_backend_desc(&p->op_graph);
  destroy_backend_desc(&p->norm_op);
  destroy_backend_desc(&p->y_desc);
  destroy_backend_desc(&p->epsilon_desc);
  destroy_backend_desc(&p->bias_desc);
  destroy_backend_desc(&p->scale_desc);
  destroy_backend_desc(&p->x_desc);
  if (p->workspace) {
    CUDA_CHECK(cudaFree(p->workspace));
    p->workspace = NULL;
  }
  if (p->dBias) {
    CUDA_CHECK(cudaFree(p->dBias));
    p->dBias = NULL;
  }
  if (p->dOut) {
    CUDA_CHECK(cudaFree(p->dOut));
    p->dOut = NULL;
  }
  if (p->dWeight) {
    CUDA_CHECK(cudaFree(p->dWeight));
    p->dWeight = NULL;
  }
  if (p->dX) {
    CUDA_CHECK(cudaFree(p->dX));
    p->dX = NULL;
  }
}

static struct rmsnorm_f32_plan *find_rmsnorm_f32_plan(int32_t N) {
  for (int i = 0; i < RMSNORM_F32_CACHE_CAP; ++i)
    if (g_rmsnorm_f32_cache[i].in_use && g_rmsnorm_f32_cache[i].N == N)
      return &g_rmsnorm_f32_cache[i];
  return NULL;
}

static struct rmsnorm_f32_plan *alloc_rmsnorm_f32_plan(int32_t N) {
  for (int i = 0; i < RMSNORM_F32_CACHE_CAP; ++i) {
    if (!g_rmsnorm_f32_cache[i].in_use) {
      memset(&g_rmsnorm_f32_cache[i], 0, sizeof(g_rmsnorm_f32_cache[i]));
      g_rmsnorm_f32_cache[i].in_use = 1;
      g_rmsnorm_f32_cache[i].N = N;
      return &g_rmsnorm_f32_cache[i];
    }
  }
  fprintf(stderr, "polygeist runtime: RMSNorm f32 cache full (cap=%d)\n",
          RMSNORM_F32_CACHE_CAP);
  abort();
}

static int build_rmsnorm_f32_plan(struct rmsnorm_f32_plan *p) {
  cudnnStatus_t last_status = CUDNN_STATUS_SUCCESS;

  p->bytes = (size_t)p->N * sizeof(float);
  p->epsilon = 1.0e-5f;
  CUDA_CHECK(cudaMalloc((void **)&p->dX, p->bytes));
  CUDA_CHECK(cudaMalloc((void **)&p->dWeight, p->bytes));
  CUDA_CHECK(cudaMalloc((void **)&p->dOut, p->bytes));
  CUDA_CHECK(cudaMalloc((void **)&p->dBias, p->bytes));
  CUDA_CHECK(cudaMemsetAsync(p->dBias, 0, p->bytes, g_stream));

  int64_t tensor_dims[4] = {1, (int64_t)p->N, 1, 1};
  int64_t tensor_strides[4] = {(int64_t)p->N, 1, 1, 1};
  int64_t scalar_dims[4] = {1, 1, 1, 1};
  int64_t scalar_strides[4] = {1, 1, 1, 1};
  int64_t uid_x = 'x';
  int64_t uid_scale = 's';
  int64_t uid_bias = 'b';
  int64_t uid_epsilon = 'e';
  int64_t uid_y = 'y';

  if (!make_f32_backend_tensor(&p->x_desc, uid_x, tensor_dims, tensor_strides, 4,
                               false, "rmsnorm.x", &last_status) ||
      !make_f32_backend_tensor(&p->scale_desc, uid_scale, tensor_dims,
                               tensor_strides, 4, false, "rmsnorm.scale",
                               &last_status) ||
      !make_f32_backend_tensor(&p->bias_desc, uid_bias, tensor_dims,
                               tensor_strides, 4, false, "rmsnorm.bias",
                               &last_status) ||
      !make_f32_backend_tensor(&p->epsilon_desc, uid_epsilon, scalar_dims,
                               scalar_strides, 4, true, "rmsnorm.epsilon",
                               &last_status) ||
      !make_f32_backend_tensor(&p->y_desc, uid_y, tensor_dims, tensor_strides, 4,
                               false, "rmsnorm.y", &last_status))
    return 0;

  last_status = cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, &p->norm_op);
  if (last_status != CUDNN_STATUS_SUCCESS) {
    report_rmsnorm_backend_fallback("rmsnorm.norm_op.create", last_status);
    return 0;
  }
  cudnnBackendNormMode_t mode = CUDNN_RMS_NORM;
  cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_INFERENCE;
  if (!set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                        CUDNN_TYPE_NORM_MODE, 1, &mode, "rmsnorm.mode",
                        &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                        CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase, "rmsnorm.phase",
                        &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->x_desc,
                        "rmsnorm.xdesc", &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->scale_desc,
                        "rmsnorm.scale_desc", &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->bias_desc,
                        "rmsnorm.bias_desc", &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->epsilon_desc,
                        "rmsnorm.epsilon_desc", &last_status) ||
      !set_backend_attr(p->norm_op, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->y_desc,
                        "rmsnorm.ydesc", &last_status) ||
      !finalize_backend_desc(p->norm_op, "rmsnorm.norm_op.finalize",
                             &last_status))
    return 0;

  last_status = cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &p->op_graph);
  if (last_status != CUDNN_STATUS_SUCCESS) {
    report_rmsnorm_backend_fallback("rmsnorm.graph.create", last_status);
    return 0;
  }
  if (!set_backend_attr(p->op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                        CUDNN_TYPE_HANDLE, 1, &g_cudnn, "rmsnorm.graph.handle",
                        &last_status) ||
      !set_backend_attr(p->op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->norm_op,
                        "rmsnorm.graph.ops", &last_status) ||
      !finalize_backend_desc(p->op_graph, "rmsnorm.graph.finalize",
                             &last_status))
    return 0;

  int64_t engine_count = 0;
  int64_t elem_count = 0;
  last_status = cudnnBackendGetAttribute(
      p->op_graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
      CUDNN_TYPE_INT64, 1, &elem_count, &engine_count);
  if (last_status != CUDNN_STATUS_SUCCESS || engine_count <= 0) {
    if (last_status == CUDNN_STATUS_SUCCESS)
      last_status = CUDNN_STATUS_NOT_SUPPORTED;
    report_rmsnorm_backend_fallback("rmsnorm.engine_count", last_status);
    return 0;
  }

  cudnnStatus_t plan_status = CUDNN_STATUS_NOT_SUPPORTED;
  for (int64_t gidx = 0; gidx < engine_count; ++gidx) {
    cudnnBackendDescriptor_t engine_tmp = NULL;
    cudnnBackendDescriptor_t cfg_tmp = NULL;
    cudnnBackendDescriptor_t plan_tmp = NULL;

    plan_status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                               &engine_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendSetAttribute(
        engine_tmp, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->op_graph);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendSetAttribute(
        engine_tmp, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1,
        &gidx);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendFinalize(engine_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;

    plan_status = cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfg_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendSetAttribute(
        cfg_tmp, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
        &engine_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendFinalize(cfg_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;

    plan_status = cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendSetAttribute(
        plan_tmp, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1,
        &g_cudnn);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendSetAttribute(
        plan_tmp, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg_tmp);
    if (plan_status != CUDNN_STATUS_SUCCESS)
      goto engine_cleanup;
    plan_status = cudnnBackendFinalize(plan_tmp);
    if (plan_status == CUDNN_STATUS_SUCCESS) {
      p->engine = engine_tmp;
      p->engine_cfg = cfg_tmp;
      p->plan = plan_tmp;
      break;
    }

engine_cleanup:
    if (plan_status == CUDNN_STATUS_SUCCESS)
      plan_status = CUDNN_STATUS_NOT_SUPPORTED;
    if (plan_tmp != p->plan)
      destroy_backend_desc(&plan_tmp);
    if (cfg_tmp != p->engine_cfg)
      destroy_backend_desc(&cfg_tmp);
    if (engine_tmp != p->engine)
      destroy_backend_desc(&engine_tmp);
  }
  if (!p->plan) {
    report_rmsnorm_backend_fallback("rmsnorm.plan", plan_status);
    return 0;
  }

  int64_t workspace_size = 0;
  last_status = cudnnBackendGetAttribute(
      p->plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1,
      &elem_count, &workspace_size);
  if (last_status != CUDNN_STATUS_SUCCESS) {
    report_rmsnorm_backend_fallback("rmsnorm.workspace_size", last_status);
    return 0;
  }
  if (workspace_size > 0)
    CUDA_CHECK(cudaMalloc(&p->workspace, (size_t)workspace_size));

  last_status = cudnnBackendCreateDescriptor(
      CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &p->variant_pack);
  if (last_status != CUDNN_STATUS_SUCCESS) {
    report_rmsnorm_backend_fallback("rmsnorm.variant.create", last_status);
    return 0;
  }
  int64_t uids[5] = {uid_x, uid_scale, uid_bias, uid_epsilon, uid_y};
  void *data_ptrs[5] = {p->dX, p->dWeight, p->dBias, &p->epsilon, p->dOut};
  if (!set_backend_attr(p->variant_pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                        CUDNN_TYPE_VOID_PTR, 5, data_ptrs,
                        "rmsnorm.variant.ptrs", &last_status) ||
      !set_backend_attr(p->variant_pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                        CUDNN_TYPE_INT64, 5, uids, "rmsnorm.variant.uids",
                        &last_status) ||
      !set_backend_attr(p->variant_pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                        CUDNN_TYPE_VOID_PTR, 1, &p->workspace,
                        "rmsnorm.variant.workspace", &last_status) ||
      !finalize_backend_desc(p->variant_pack, "rmsnorm.variant.finalize",
                             &last_status))
    return 0;

  return 1;
}

static struct rmsnorm_f32_plan *get_rmsnorm_f32_plan(int32_t N) {
  struct rmsnorm_f32_plan *p = find_rmsnorm_f32_plan(N);
  if (p) return p;

  p = alloc_rmsnorm_f32_plan(N);
  if (!build_rmsnorm_f32_plan(p)) {
    release_rmsnorm_f32_plan_resources(p);
    p->unsupported = 1;
  }
  return p;
}

static int try_cudnn_rmsnorm_f32(
    int32_t N, const float *X, const float *Weight, float *Out,
    double host_start_ms) {
  struct rmsnorm_f32_plan *p = get_rmsnorm_f32_plan(N);
  if (!p || p->unsupported)
    return 0;

  CUDA_CHECK(cudaMemcpyAsync(p->dX, X, p->bytes, cudaMemcpyHostToDevice,
                             g_stream));
  CUDA_CHECK(cudaMemcpyAsync(p->dWeight, Weight, p->bytes,
                             cudaMemcpyHostToDevice, g_stream));

  timing_gpu_begin();
  CUDNN_CHECK(cudnnBackendExecute(g_cudnn, p->plan, p->variant_pack));
  CUDA_CHECK(cudaMemcpyAsync(Out, p->dOut, p->bytes, cudaMemcpyDeviceToHost,
                             g_stream));
  timing_gpu_end("cudnnRmsNormForward", 1, N, 0, host_start_ms);
  return 1;
}

void polygeist_rmsnorm_f32(
    int32_t N, const float *X, const float *Weight, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  if (try_cudnn_rmsnorm_f32(N, X, Weight, Out, host_start_ms))
    return;

  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  rmsnorm_host_f32(N, X, Weight, Out);

  timing_host_only("host_rmsnorm_f32", N, 1, 0, host_start_ms);
}

void polygeist_cudnn_softmax_forward_f32(int32_t N, float *X) {
  if (N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe(X, bytes);

  cudnnTensorDescriptor_t x_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, 1, N));

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnSoftmaxForward(
      g_cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
      &alpha, x_desc, dX, &beta, x_desc, dX));
  timing_gpu_end("cudnnSoftmaxForward", 1, N, 0, host_start_ms);

  cudnnDestroyTensorDescriptor(x_desc);
}

void polygeist_cudnn_softmax_forward_out_f32(
    int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);

  timing_gpu_begin();
  CUDA_CHECK(cudaMemcpyAsync(dOut, dX, bytes, cudaMemcpyDeviceToDevice,
                             g_stream));
  timing_gpu_end("cudaCopySoftmaxInput_f32", N, 1, 0, host_start_ms);
  polygeist_cudnn_softmax_forward_f32(N, Out);
}

void polygeist_cuda_copy_f32(int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);

  timing_gpu_begin();
  CUDA_CHECK(cudaMemcpyAsync(dOut, dX, bytes, cudaMemcpyDeviceToDevice,
                             g_stream));
  timing_gpu_end("cudaCopy_f32", N, 1, 0, host_start_ms);
}

void polygeist_cuda_add_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dY = (float *)register_host_safe((void *)Y, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);
  const float alpha = 1.0f;

  timing_gpu_begin();
  CUDA_CHECK(cudaMemcpyAsync(dOut, dX, bytes, cudaMemcpyDeviceToDevice,
                             g_stream));
  CUBLAS_CHECK(cublasSaxpy(g_handle, N, &alpha, dY, 1, dOut, 1));
  timing_gpu_end("cudaAdd_f32", N, 1, 0, host_start_ms);
}

void polygeist_cuda_mask_select_f32(
    int32_t N, int32_t pos, const float *Scores, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *keep_h = (float *)malloc(bytes);
  float *bias_h = (float *)malloc(bytes);
  if (!keep_h || !bias_h) {
    fprintf(stderr, "polygeist_cuda_mask_select_f32: malloc failed\n");
    abort();
  }
  for (int32_t i = 0; i < N; ++i) {
    int drop = i > pos;
    keep_h[i] = drop ? 0.0f : 1.0f;
    bias_h[i] = drop ? -3.4028234663852886e38f : 0.0f;
  }

  float *dScores = (float *)register_host_safe((void *)Scores, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);
  float *dKeep = NULL;
  float *dBias = NULL;
  CUDA_CHECK(cudaMalloc((void **)&dKeep, bytes));
  CUDA_CHECK(cudaMalloc((void **)&dBias, bytes));

  cudnnTensorDescriptor_t desc;
  cudnnOpTensorDescriptor_t mul_desc, add_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, 1, N));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_desc));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&add_desc));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      mul_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      add_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

  float one = 1.0f;
  float zero = 0.0f;
  timing_gpu_begin();
  CUDA_CHECK(cudaMemcpyAsync(dKeep, keep_h, bytes, cudaMemcpyHostToDevice,
                             g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dBias, bias_h, bytes, cudaMemcpyHostToDevice,
                             g_stream));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, desc, dScores,
                            &one, desc, dKeep,
                            &zero, desc, dOut));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, add_desc,
                            &one, desc, dOut,
                            &one, desc, dBias,
                            &zero, desc, dOut));
  timing_gpu_end("cudaMaskSelect_f32", N, 1, 0, host_start_ms);

  cudnnDestroyOpTensorDescriptor(mul_desc);
  cudnnDestroyOpTensorDescriptor(add_desc);
  cudnnDestroyTensorDescriptor(desc);
  cudaFree(dKeep);
  cudaFree(dBias);
  free(keep_h);
  free(bias_h);
}

void polygeist_cuda_swiglu_f32(
    int32_t N, const float *Gate, const float *Up, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dGate = (float *)register_host_safe((void *)Gate, bytes);
  float *dUp = (float *)register_host_safe((void *)Up, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);
  float *dSigmoid = NULL;
  CUDA_CHECK(cudaMalloc((void **)&dSigmoid, bytes));

  cudnnTensorDescriptor_t desc;
  cudnnActivationDescriptor_t act_desc;
  cudnnOpTensorDescriptor_t mul_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, 1, N));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_desc));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      mul_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

  float one = 1.0f;
  float zero = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnActivationForward(
      g_cudnn, act_desc, &one, desc, dGate, &zero, desc, dSigmoid));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, desc, dGate,
                            &one, desc, dSigmoid,
                            &zero, desc, dOut));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, desc, dOut,
                            &one, desc, dUp,
                            &zero, desc, dOut));
  timing_gpu_end("cudaSwiGLU_f32", N, 1, 0, host_start_ms);

  cudnnDestroyOpTensorDescriptor(mul_desc);
  cudnnDestroyActivationDescriptor(act_desc);
  cudnnDestroyTensorDescriptor(desc);
  cudaFree(dSigmoid);
}

void polygeist_cuda_rope_mulmul_f32(
    int32_t M, int32_t N, const float *A, const float *B,
    const float *C, const float *D, float *Out, int32_t add) {
  if (M <= 0 || N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t mat_bytes = (size_t)M * (size_t)N * sizeof(float);
  size_t vec_bytes = (size_t)N * sizeof(float);
  float *dA = (float *)register_host_safe((void *)A, mat_bytes);
  float *dB = (float *)register_host_safe((void *)B, vec_bytes);
  float *dC = (float *)register_host_safe((void *)C, mat_bytes);
  float *dD = (float *)register_host_safe((void *)D, vec_bytes);
  float *dOut = (float *)register_host_safe(Out, mat_bytes);
  float *dTmp = NULL;
  CUDA_CHECK(cudaMalloc((void **)&dTmp, mat_bytes));

  cudnnTensorDescriptor_t mat_desc, vec_desc;
  cudnnOpTensorDescriptor_t mul_desc, add_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&mat_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&vec_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(mat_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(vec_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, 1, N));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_desc));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&add_desc));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      mul_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      add_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

  float one = 1.0f;
  float zero = 0.0f;
  float sign = add ? 1.0f : -1.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, mat_desc, dA,
                            &one, vec_desc, dB,
                            &zero, mat_desc, dOut));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, mat_desc, dC,
                            &one, vec_desc, dD,
                            &zero, mat_desc, dTmp));
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, add_desc,
                            &one, mat_desc, dOut,
                            &sign, mat_desc, dTmp,
                            &zero, mat_desc, dOut));
  timing_gpu_end(add ? "cudaRopeMulMulAdd_f32" : "cudaRopeMulMulSub_f32",
                 M, N, 0, host_start_ms);

  cudnnDestroyOpTensorDescriptor(mul_desc);
  cudnnDestroyOpTensorDescriptor(add_desc);
  cudnnDestroyTensorDescriptor(mat_desc);
  cudnnDestroyTensorDescriptor(vec_desc);
  cudaFree(dTmp);
}

void polygeist_cublas_time_begin(void) {
  polygeist_cublas_init();
  cudaEventRecord(g_ev_begin, g_stream);
}

double polygeist_cublas_time_end_ms(void) {
  cudaEventRecord(g_ev_end, g_stream);
  cudaEventSynchronize(g_ev_end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, g_ev_begin, g_ev_end);
  return (double)ms;
}
