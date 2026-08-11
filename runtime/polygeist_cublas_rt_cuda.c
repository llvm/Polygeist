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
#if defined(__has_include)
#  if __has_include(<cufft.h>)
#    include <cufft.h>
#    define POLYGEIST_HAS_CUFFT 1
#  endif
#endif
#ifndef POLYGEIST_HAS_CUFFT
#  define POLYGEIST_HAS_CUFFT 0
#endif
#if defined(POLYGEIST_ENABLE_CUTENSORNET)
#  include <cutensornet.h>
#  define POLYGEIST_HAS_CUTENSORNET 1
#else
#  define POLYGEIST_HAS_CUTENSORNET 0
#endif
#if defined(POLYGEIST_ENABLE_CUTENSOR)
#  include <cutensor.h>
#  define POLYGEIST_HAS_CUTENSOR 1
#else
#  define POLYGEIST_HAS_CUTENSOR 0
#endif
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
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
static int            g_atexit_registered = 0;
static int            g_pipeline_depth = 0;
static int            g_timing_enabled = -1;
static FILE          *g_timing_file = NULL;

typedef struct {
  void *ptr;
  size_t bytes;
  int in_use;
} DeviceTempEntry;

static DeviceTempEntry *g_device_temps = NULL;
static size_t g_device_temp_count = 0;
static size_t g_device_temp_cap = 0;

static void **g_deferred_device_frees = NULL;
static size_t g_deferred_device_free_count = 0;
static size_t g_deferred_device_free_cap = 0;

static void **g_deferred_host_frees = NULL;
static size_t g_deferred_host_free_count = 0;
static size_t g_deferred_host_free_cap = 0;

#if POLYGEIST_HAS_CUTENSORNET
static void destroy_cutensornet_contraction_cache(void);
#endif

#if POLYGEIST_HAS_CUTENSOR
#define CUTENSOR_CHECK(call) do {                                            \
    cutensorStatus_t s = (call);                                             \
    if (s != CUTENSOR_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "%s:%d cuTENSOR error: %d (%s)\n", __FILE__,         \
              __LINE__, (int)s, cutensorGetErrorString(s));                  \
      abort();                                                               \
    }                                                                        \
  } while (0)
#endif

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

#if POLYGEIST_HAS_CUFFT
#define CUFFT_CHECK(call) do {                                               \
    cufftResult s = (call);                                                  \
    if (s != CUFFT_SUCCESS) {                                                \
      fprintf(stderr, "%s:%d cufft error: %d\n", __FILE__, __LINE__,         \
              (int)s);                                                       \
      abort();                                                               \
    }                                                                        \
  } while (0)
#endif

#if POLYGEIST_HAS_CUTENSORNET
#define CUTENSORNET_CHECK(call) do {                                         \
    cutensornetStatus_t s = (call);                                          \
    if (s != CUTENSORNET_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s:%d cuTensorNet error: %s\n", __FILE__, __LINE__, \
              cutensornetGetErrorString(s));                                 \
      abort();                                                               \
    }                                                                        \
  } while (0)
#endif

static int in_pipeline_scope(void) { return g_pipeline_depth > 0; }

static void sync_stream_if_outside_pipeline(void) {
  if (!in_pipeline_scope())
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

static void reserve_device_temp_entries(size_t need) {
  if (need <= g_device_temp_cap)
    return;
  size_t new_cap = g_device_temp_cap ? g_device_temp_cap * 2 : 16;
  while (new_cap < need)
    new_cap *= 2;
  DeviceTempEntry *next =
      (DeviceTempEntry *)realloc(g_device_temps,
                                 new_cap * sizeof(DeviceTempEntry));
  if (!next) {
    fprintf(stderr, "polygeist runtime: device temp cache realloc failed\n");
    abort();
  }
  g_device_temps = next;
  g_device_temp_cap = new_cap;
}

static void *pipeline_device_malloc(size_t bytes) {
  if (bytes == 0)
    return NULL;

  if (in_pipeline_scope()) {
    ssize_t best = -1;
    for (size_t i = 0; i < g_device_temp_count; ++i) {
      if (g_device_temps[i].in_use || g_device_temps[i].bytes < bytes)
        continue;
      if (best < 0 || g_device_temps[i].bytes < g_device_temps[best].bytes)
        best = (ssize_t)i;
    }
    if (best >= 0) {
      g_device_temps[best].in_use = 1;
      return g_device_temps[best].ptr;
    }
  }

  void *ptr = NULL;
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  if (!in_pipeline_scope())
    return ptr;

  reserve_device_temp_entries(g_device_temp_count + 1);
  g_device_temps[g_device_temp_count].ptr = ptr;
  g_device_temps[g_device_temp_count].bytes = bytes;
  g_device_temps[g_device_temp_count].in_use = 1;
  g_device_temp_count++;
  return ptr;
}

static ssize_t find_device_temp(void *ptr) {
  for (size_t i = 0; i < g_device_temp_count; ++i)
    if (g_device_temps[i].ptr == ptr)
      return (ssize_t)i;
  return -1;
}

static void reserve_deferred_device_frees(size_t need) {
  if (need <= g_deferred_device_free_cap)
    return;
  size_t new_cap =
      g_deferred_device_free_cap ? g_deferred_device_free_cap * 2 : 16;
  while (new_cap < need)
    new_cap *= 2;
  void **next =
      (void **)realloc(g_deferred_device_frees, new_cap * sizeof(void *));
  if (!next) {
    fprintf(stderr, "polygeist runtime: deferred device-free realloc failed\n");
    abort();
  }
  g_deferred_device_frees = next;
  g_deferred_device_free_cap = new_cap;
}

static void flush_deferred_device_frees(void) {
  for (size_t i = 0; i < g_deferred_device_free_count; ++i)
    CUDA_CHECK(cudaFree(g_deferred_device_frees[i]));
  g_deferred_device_free_count = 0;
}

static void destroy_deferred_device_free_list(void) {
  flush_deferred_device_frees();
  free(g_deferred_device_frees);
  g_deferred_device_frees = NULL;
  g_deferred_device_free_cap = 0;
}

static void pipeline_device_free(void *ptr) {
  if (!ptr)
    return;
  ssize_t idx = find_device_temp(ptr);
  if (idx >= 0) {
    g_device_temps[idx].in_use = 0;
    return;
  }
  if (in_pipeline_scope()) {
    reserve_deferred_device_frees(g_deferred_device_free_count + 1);
    g_deferred_device_frees[g_deferred_device_free_count++] = ptr;
    return;
  }
  CUDA_CHECK(cudaFree(ptr));
}

static void destroy_device_temp_cache(void) {
  for (size_t i = 0; i < g_device_temp_count; ++i)
    if (g_device_temps[i].ptr)
      CUDA_CHECK(cudaFree(g_device_temps[i].ptr));
  free(g_device_temps);
  g_device_temps = NULL;
  g_device_temp_count = 0;
  g_device_temp_cap = 0;
}

static void reserve_deferred_host_frees(size_t need) {
  if (need <= g_deferred_host_free_cap)
    return;
  size_t new_cap = g_deferred_host_free_cap ? g_deferred_host_free_cap * 2 : 16;
  while (new_cap < need)
    new_cap *= 2;
  void **next =
      (void **)realloc(g_deferred_host_frees, new_cap * sizeof(void *));
  if (!next) {
    fprintf(stderr, "polygeist runtime: deferred host-free realloc failed\n");
    abort();
  }
  g_deferred_host_frees = next;
  g_deferred_host_free_cap = new_cap;
}

static void pipeline_host_free(void *ptr) {
  if (!ptr)
    return;
  if (!in_pipeline_scope()) {
    free(ptr);
    return;
  }
  reserve_deferred_host_frees(g_deferred_host_free_count + 1);
  g_deferred_host_frees[g_deferred_host_free_count++] = ptr;
}

static void flush_deferred_host_frees(void) {
  for (size_t i = 0; i < g_deferred_host_free_count; ++i)
    free(g_deferred_host_frees[i]);
  g_deferred_host_free_count = 0;
}

static void destroy_deferred_host_free_list(void) {
  flush_deferred_host_frees();
  free(g_deferred_host_frees);
  g_deferred_host_frees = NULL;
  g_deferred_host_free_cap = 0;
}

#define DEVICE_MALLOC(ptrptr, bytes)                                         \
  do {                                                                       \
    *(void **)(ptrptr) = pipeline_device_malloc((size_t)(bytes));             \
  } while (0)

#define DEVICE_FREE(ptr) pipeline_device_free((void *)(ptr))

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

static void timing_host_only(
    const char *op, int32_t m, int32_t n, int32_t k, double host_start_ms);

static void timing_gpu_begin(void) {
  if (timing_enabled() && !in_pipeline_scope())
    CUDA_CHECK(cudaEventRecord(g_ev_begin, g_stream));
}

static void timing_gpu_end(
    const char *op, int32_t m, int32_t n, int32_t k, double host_start_ms) {
  if (in_pipeline_scope()) {
    timing_host_only(op, m, n, k, host_start_ms);
    return;
  }
  if (!timing_enabled()) {
    sync_stream_if_outside_pipeline();
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
// Cache implementation: a small linear table keyed on host range.  Larger
// raised application graphs can create more than 256 temporary tensor
// snapshots over their lifetime, so a full cache evicts its least-recently
// used registration instead of aborting.  Calls synchronize the CUDA stream
// before returning, which makes an entry from an earlier call safe to evict;
// a still-live tensor is simply registered again on its next use.

#define HOSTREG_CACHE_CAP 256
struct hostreg_entry {
  void *host;
  void *dev;
  size_t bytes;
  uint64_t last_use;
};
static struct hostreg_entry g_hostreg_cache[HOSTREG_CACHE_CAP];
static int g_hostreg_count = 0;
static uint64_t g_hostreg_clock = 0;

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
      e->last_use = ++g_hostreg_clock;
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
    int victim = 0;
    for (int i = 1; i < g_hostreg_count; ++i) {
      if (g_hostreg_cache[i].last_use < g_hostreg_cache[victim].last_use)
        victim = i;
    }
    cudaError_t err = cudaHostUnregister(g_hostreg_cache[victim].host);
    if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
      fprintf(stderr, "%s:%d cudaHostUnregister(%p) failed: %s\n",
              __FILE__, __LINE__, g_hostreg_cache[victim].host,
              cudaGetErrorString(err));
      abort();
    }
    g_hostreg_cache[victim].host = host;
    g_hostreg_cache[victim].dev = dev;
    g_hostreg_cache[victim].bytes = bytes;
    g_hostreg_cache[victim].last_use = ++g_hostreg_clock;
    return;
  }
  g_hostreg_cache[g_hostreg_count].host = host;
  g_hostreg_cache[g_hostreg_count].dev  = dev;
  g_hostreg_cache[g_hostreg_count].bytes = bytes;
  g_hostreg_cache[g_hostreg_count].last_use = ++g_hostreg_clock;
  g_hostreg_count++;
}

// We tried bypassing cudaHostRegister and passing host pointers directly
// to cuBLAS — fails with illegal-memory-access. cuBLAS requires the
// buffer to be registered (or device-allocated) even on a Tegra SoC
// where the iGPU can technically reach any DRAM page.
static int pointer_is_device_resident(void *ptr, void **device_ptr) {
  struct cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    // Unregistered malloc pointers normally report cudaErrorInvalidValue.
    // Clear the sticky runtime error before the registration path below.
    (void)cudaGetLastError();
    return 0;
  }
#if CUDART_VERSION >= 10000
  if (attrs.type != cudaMemoryTypeDevice &&
      attrs.type != cudaMemoryTypeManaged)
    return 0;
#else
  if (attrs.memoryType != cudaMemoryTypeDevice)
    return 0;
#endif
  *device_ptr = attrs.devicePointer ? attrs.devicePointer : ptr;
  return 1;
}

static void *register_host_safe(void *ptr, size_t bytes) {
  void *cached = hostreg_cache_lookup(ptr, bytes);
  if (cached) return cached;
  void *device_ptr = NULL;
  if (pointer_is_device_resident(ptr, &device_ptr))
    return device_ptr;
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
  // Register after CUDA has initialized its own process-exit hooks. atexit
  // runs in reverse order, so our cache/stream teardown happens first.
  if (!g_atexit_registered) {
    if (atexit(polygeist_cublas_destroy) != 0) {
      fprintf(stderr, "polygeist runtime: failed to register CUDA cleanup\n");
      abort();
    }
    g_atexit_registered = 1;
  }
}

void polygeist_cublas_destroy(void) {
  if (g_timing_file && g_timing_file != stderr) {
    fclose(g_timing_file);
    g_timing_file = NULL;
  }
  if (!g_initialized) return;
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
#if POLYGEIST_HAS_CUTENSORNET
  destroy_cutensornet_contraction_cache();
#endif
  destroy_deferred_device_free_list();
  destroy_deferred_host_free_list();
  destroy_device_temp_cache();
  cudaEventDestroy(g_ev_begin);
  cudaEventDestroy(g_ev_end);
  cublasDestroy(g_handle);
  cudaStreamDestroy(g_stream);
  g_initialized = 0;
  g_pipeline_depth = 0;
}

void polygeist_cublas_pipeline_begin(void) {
  polygeist_cublas_init();
  g_pipeline_depth++;
}

void polygeist_cublas_pipeline_end(void) {
  if (!g_initialized)
    return;
  if (g_pipeline_depth > 0)
    g_pipeline_depth--;
  if (g_pipeline_depth == 0) {
    sync_stream_if_outside_pipeline();
    flush_deferred_device_frees();
    flush_deferred_host_frees();
  }
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

void polygeist_cublas_sgemm_transpose(
    int32_t M, int32_t N, int32_t K,
    int32_t transA, int32_t transB,
    float alpha,
    const float *A, int32_t lda,
    const float *B, int32_t ldb,
    float beta,
    float *C, int32_t ldc) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  int32_t aRows = transA ? K : M;
  int32_t bRows = transB ? N : K;
  size_t bytes_A = (size_t)aRows * (size_t)lda * sizeof(float);
  size_t bytes_B = (size_t)bRows * (size_t)ldb * sizeof(float);
  size_t bytes_C = (size_t)M * (size_t)ldc * sizeof(float);
  float *dA = (float *)register_host_safe((void *)A, bytes_A);
  float *dB = (float *)register_host_safe((void *)B, bytes_B);
  float *dC = (float *)register_host_safe(C, bytes_C);
  timing_gpu_begin();
  // Row-major C=op(A)op(B) becomes column-major C^T=op(B)^T op(A)^T.
  CUBLAS_CHECK(cublasSgemm(g_handle,
                           transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                           transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                           N, M, K, &alpha, dB, ldb, dA, lda, &beta,
                           dC, ldc));
  timing_gpu_end("cublasSgemm_transpose", M, N, K, host_start_ms);
  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

void polygeist_cublas_sgemm_strided_batched_broadcast_rhs(
    int32_t batch, int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, float *C) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t stride_A = (size_t)M * (size_t)K;
  size_t stride_C = (size_t)M * (size_t)N;
  float *dA = (float *)register_host_safe(
      (void *)A, (size_t)batch * stride_A * sizeof(float));
  float *dB = (float *)register_host_safe(
      (void *)B, (size_t)K * (size_t)N * sizeof(float));
  float *dC = (float *)register_host_safe(
      C, (size_t)batch * stride_C * sizeof(float));
  const float one = 1.0f;
  const float zero = 0.0f;

  // Row-major C_b=A_b*B is column-major C_b^T=B^T*A_b^T.  A zero
  // stride for B broadcasts the same right-hand matrix to every batch.
  timing_gpu_begin();
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
      /*m=*/N, /*n=*/M, /*k=*/K,
      &one,
      dB, /*ldb=*/N, /*strideB=*/0,
      dA, /*lda=*/K, /*strideA=*/(long long)stride_A,
      &zero,
      dC, /*ldc=*/N, /*strideC=*/(long long)stride_C,
      batch));
  timing_gpu_end("cublasSgemmStridedBatched_broadcast_rhs",
                 batch * M, N, K, host_start_ms);

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

void polygeist_cublas_sgemm_strided_batched(
    int32_t batch, int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, float *C) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t strideA = (size_t)M * K, strideB = (size_t)K * N;
  size_t strideC = (size_t)M * N;
  float *dA = (float *)register_host_safe((void *)A, batch * strideA * sizeof(float));
  float *dB = (float *)register_host_safe((void *)B, batch * strideB * sizeof(float));
  float *dC = (float *)register_host_safe(C, batch * strideC * sizeof(float));
  float one = 1.0f, zero = 0.0f;
  timing_gpu_begin();
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
      dB, N, (long long)strideB, dA, K, (long long)strideA,
      &zero, dC, N, (long long)strideC, batch));
  timing_gpu_end("cublasSgemmStridedBatched", batch * M, N, K, host_start_ms);
  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

void polygeist_cublas_dgemm_outer_product(
    int32_t M, int32_t N,
    const double *u, const double *v, double *C) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  double *du = (double *)register_host_safe(
      (void *)u, (size_t)M * sizeof(double));
  double *dv = (double *)register_host_safe(
      (void *)v, (size_t)N * sizeof(double));
  double *dC = (double *)register_host_safe(
      C, (size_t)M * (size_t)N * sizeof(double));
  const double one = 1.0;
  const double zero = 0.0;

  // C(row-major MxN)^T = v(Nx1) * u^T(1xM).  beta=0 gives overwrite
  // semantics without a separate zero-fill launch.
  timing_gpu_begin();
  CUBLAS_CHECK(cublasDgemm(g_handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m=*/N, /*n=*/M, /*k=*/1,
                           &one,
                           dv, /*lda=*/N,
                           du, /*ldb=*/1,
                           &zero,
                           dC, /*ldc=*/N));
  timing_gpu_end("cublasDgemm_outer_product", M, N, 1, host_start_ms);

  unregister_host_safe((void *)u);
  unregister_host_safe((void *)v);
  unregister_host_safe(C);
}

// Zero mapped host buffers on the CPU, but preserve device residency when the
// caller supplies a cudaMalloc/cudaMallocManaged allocation.
void polygeist_cublas_memset_zero_2d(int32_t M, int32_t N,
                                       double *A, int32_t lda) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  void *device_ptr = NULL;
  if (pointer_is_device_resident(A, &device_ptr)) {
    polygeist_cublas_init();
    timing_gpu_begin();
    CUDA_CHECK(cudaMemset2DAsync(device_ptr, (size_t)lda * sizeof(double), 0,
                                 (size_t)N * sizeof(double), M, g_stream));
    timing_gpu_end("cuda_memset_zero_2d_f64", M, N, 0, host_start_ms);
    return;
  }
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

void polygeist_cublas_saxpby(int32_t N, float alpha, const float *x,
                              float beta, float *y) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)N * sizeof(float);
  float *dx = (float *)register_host_safe((void *)x, bytes);
  float *dy = (float *)register_host_safe(y, bytes);
  timing_gpu_begin();
  CUBLAS_CHECK(cublasSscal(g_handle, N, &beta, dy, 1));
  CUBLAS_CHECK(cublasSaxpy(g_handle, N, &alpha, dx, 1, dy, 1));
  timing_gpu_end("cublasSaxpby", N, 1, 0, host_start_ms);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

void polygeist_cublas_sscal(int32_t N, float scale, float *x) {
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)N * sizeof(float);
  float *dx = (float *)register_host_safe(x, bytes);
  timing_gpu_begin();
  CUBLAS_CHECK(cublasSscal(g_handle, N, &scale, dx, 1));
  timing_gpu_end("cublasSscal", N, 1, 0, host_start_ms);
  unregister_host_safe(x);
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

// Preserve the zero-copy host behavior for ordinary C allocations, but also
// accept an already-device-resident buffer.  This lets a lifted function use
// the exact same ABI with cudaMalloc operands without attempting a CPU memset
// through a device address.
void polygeist_cublas_memset_zero_1d(int32_t N, double *v) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  void *device_ptr = NULL;
  if (pointer_is_device_resident(v, &device_ptr)) {
    polygeist_cublas_init();
    timing_gpu_begin();
    CUDA_CHECK(cudaMemsetAsync(device_ptr, 0, (size_t)N * sizeof(double),
                               g_stream));
    timing_gpu_end("cuda_memset_zero_1d_f64", N, 1, 0, host_start_ms);
    return;
  }
  memset(v, 0, (size_t)N * sizeof(double));
  timing_host_only("host_memset_zero_1d_f64", N, 1, 0, host_start_ms);
}

void polygeist_cublas_memset_zero_1d_f32(int32_t N, float *v) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  void *device_ptr = NULL;
  if (pointer_is_device_resident(v, &device_ptr)) {
    polygeist_cublas_init();
    timing_gpu_begin();
    CUDA_CHECK(cudaMemsetAsync(device_ptr, 0, (size_t)N * sizeof(float),
                               g_stream));
    timing_gpu_end("cuda_memset_zero_1d_f32", N, 1, 0, host_start_ms);
    return;
  }
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
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

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
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
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
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

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
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv2d_5x5_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2, double w3, double w4,
    double w5, double w6, double w7, double w8, double w9,
    double w10, double w11, double w12, double w13, double w14,
    double w15, double w16, double w17, double w18, double w19,
    double w20, double w21, double w22, double w23, double w24,
    const double *A, double *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  const double filter_h[25] = {
      w0, w1, w2, w3, w4, w5, w6, w7, w8, w9,
      w10, w11, w12, w13, w14, w15, w16, w17, w18, w19,
      w20, w21, w22, w23, w24};

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_DOUBLE,
                                          CUDNN_TENSOR_NCHW, 1, 1, 5, 5));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, M - 4, N - 4));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(double);
  size_t bytes_f   = 25 * sizeof(double);
  size_t bytes_out = (size_t)(M - 4) * (size_t)(N - 4) * sizeof(double);
  double *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f64 5x5): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  double alpha = 1.0, beta = 0.0;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_25tap_f64", M, N, 25, host_start_ms);

  for (int32_t i = 0; i < M - 4; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)i * (size_t)N,
        dB + (size_t)i * (size_t)(N - 4),
        (size_t)(N - 4) * sizeof(double),
        cudaMemcpyDeviceToHost, g_stream));
  }
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv2d_5x5_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2, float w3, float w4,
    float w5, float w6, float w7, float w8, float w9,
    float w10, float w11, float w12, float w13, float w14,
    float w15, float w16, float w17, float w18, float w19,
    float w20, float w21, float w22, float w23, float w24,
    const float *A, float *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  const float filter_h[25] = {
      w0, w1, w2, w3, w4, w5, w6, w7, w8, w9,
      w10, w11, w12, w13, w14, w15, w16, w17, w18, w19,
      w20, w21, w22, w23, w24};

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
                                          CUDNN_TENSOR_NCHW, 1, 1, 5, 5));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, M - 4, N - 4));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(float);
  size_t bytes_f   = 25 * sizeof(float);
  size_t bytes_out = (size_t)(M - 4) * (size_t)(N - 4) * sizeof(float);
  float *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, filter_h, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f32 5x5): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_25tap_f32", M, N, 25, host_start_ms);

  for (int32_t i = 0; i < M - 4; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)i * (size_t)N,
        dB + (size_t)i * (size_t)(N - 4),
        (size_t)(N - 4) * sizeof(float),
        cudaMemcpyDeviceToHost, g_stream));
  }
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv2d_ntap_f64(
    int32_t M, int32_t N, int32_t K,
    const double *W, const double *A, double *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  int32_t out_h = M - (K - 1);
  int32_t out_w = N - (K - 1);
  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, M, N));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(f_desc, CUDNN_DATA_DOUBLE,
                                          CUDNN_TENSOR_NCHW, 1, 1, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE, 1, 1, out_h, out_w));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(double);
  size_t bytes_f   = (size_t)K * (size_t)K * sizeof(double);
  size_t bytes_out = (size_t)out_h * (size_t)out_w * sizeof(double);
  double *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, W, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f64 ntap): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  double alpha = 1.0, beta = 0.0;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_ntap_f64", M, N, K * K, host_start_ms);

  for (int32_t i = 0; i < out_h; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)i * (size_t)N,
        dB + (size_t)i * (size_t)out_w,
        (size_t)out_w * sizeof(double),
        cudaMemcpyDeviceToHost, g_stream));
  }
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv2d_ntap_f32(
    int32_t M, int32_t N, int32_t K,
    const float *W, const float *A, float *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  int32_t out_h = M - (K - 1);
  int32_t out_w = N - (K - 1);
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
                                          CUDNN_TENSOR_NCHW, 1, 1, K, K));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, out_h, out_w));

  size_t bytes_in  = (size_t)M * (size_t)N * sizeof(float);
  size_t bytes_f   = (size_t)K * (size_t)K * sizeof(float);
  size_t bytes_out = (size_t)out_h * (size_t)out_w * sizeof(float);
  float *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, W, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f32 ntap): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution2D_ntap_f32", M, N, K * K, host_start_ms);

  for (int32_t i = 0; i < out_h; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
        B + (size_t)i * (size_t)N,
        dB + (size_t)i * (size_t)out_w,
        (size_t)out_w * sizeof(float),
        cudaMemcpyDeviceToHost, g_stream));
  }
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv3d_ntap_f64(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const double *W, const double *A, double *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  int in_dims[5] = {1, 1, inD, inH, inW};
  int in_strides[5] = {
      inD * inH * inW, inD * inH * inW, inH * inW, inW, 1};
  int out_dims[5] = {1, 1, outD, outH, outW};
  int out_strides[5] = {
      outD * outH * outW, outD * outH * outW, outH * outW, outW, 1};
  int filt_dims[5] = {1, 1, K, K, K};
  int pad[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  int dilation[3] = {1, 1, 1};

  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      in_desc, CUDNN_DATA_DOUBLE, 5, in_dims, in_strides));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      f_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 5, filt_dims));
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      conv_desc, 3, pad, stride, dilation,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      out_desc, CUDNN_DATA_DOUBLE, 5, out_dims, out_strides));

  size_t bytes_in = (size_t)inD * (size_t)inH * (size_t)inW * sizeof(double);
  size_t bytes_f = (size_t)K * (size_t)K * (size_t)K * sizeof(double);
  size_t bytes_out =
      (size_t)outD * (size_t)outH * (size_t)outW * sizeof(double);
  double *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, W, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f64 conv3d ntap): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  double alpha = 1.0, beta = 0.0;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution3D_ntap_f64",
                 outD, outH * outW, K * K * K, host_start_ms);

  CUDA_CHECK(cudaMemcpyAsync(B, dB, bytes_out, cudaMemcpyDeviceToHost, g_stream));
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv3d_ntap_f32(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const float *W, const float *A, float *B) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  ensure_cudnn();

  cudnnTensorDescriptor_t      in_desc, out_desc;
  cudnnFilterDescriptor_t      f_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&f_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  int in_dims[5] = {1, 1, inD, inH, inW};
  int in_strides[5] = {
      inD * inH * inW, inD * inH * inW, inH * inW, inW, 1};
  int out_dims[5] = {1, 1, outD, outH, outW};
  int out_strides[5] = {
      outD * outH * outW, outD * outH * outW, outH * outW, outW, 1};
  int filt_dims[5] = {1, 1, K, K, K};
  int pad[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  int dilation[3] = {1, 1, 1};

  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      in_desc, CUDNN_DATA_FLOAT, 5, in_dims, in_strides));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      f_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filt_dims));
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      conv_desc, 3, pad, stride, dilation,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      out_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides));

  size_t bytes_in = (size_t)inD * (size_t)inH * (size_t)inW * sizeof(float);
  size_t bytes_f = (size_t)K * (size_t)K * (size_t)K * sizeof(float);
  size_t bytes_out =
      (size_t)outD * (size_t)outH * (size_t)outW * sizeof(float);
  float *dA = NULL, *dF = NULL, *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_in, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dF, W, bytes_f, cudaMemcpyHostToDevice, g_stream));

  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  int n_returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, 1, &n_returned, &algo_perf));
  if (n_returned < 1) {
    fprintf(stderr, "cuDNN(f32 conv3d ntap): no fwd algo available\n");
    abort();
  }

  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, in_desc, f_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));
  void *dWS = NULL;
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));
  timing_gpu_end("cudnnConvolution3D_ntap_f32",
                 outD, outH * outW, K * K * K, host_start_ms);

  CUDA_CHECK(cudaMemcpyAsync(B, dB, bytes_out, cudaMemcpyDeviceToHost, g_stream));
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

extern void polygeist_custom_stencil3d_7pt_flat_f64_device(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6,
    void *cuda_stream) __attribute__((weak));

extern void polygeist_custom_stencil3d_7pt_flat_f32_device(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6,
    void *cuda_stream) __attribute__((weak));

static void polygeist_custom_stencil3d_7pt_flat_f64_cpu(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  for (int32_t i = 0; i < N; ++i) {
    double extra_v = extra ? extra[i] : 0.0;
    double scale = coeff ? coeff[i] : 1.0;
    double base = base0 * a0[i] + (extra ? base_extra * extra_v : 0.0);
    double inner = c0 * a0[i] + c1 * a1[i] + c2 * a2[i] +
                   c3 * a3[i] + c4 * a4[i] + c5 * a5[i] +
                   c6 * a6[i] + (extra ? coeff_extra * extra_v : 0.0);
    out[i] = base + scale * inner;
  }
  if (timing_enabled()) {
    fprintf(timing_file(),
            "POLYGEIST_RT_TIMING\top=customStencil3D7pt_f64_cpu_fallback"
            "\tm=%d\tn=1\tk=7\thost_ms=%.6f\tdevice_ms=0.000000\n",
            N, wall_time_ms() - host_start_ms);
    fflush(timing_file());
  }
}

static void polygeist_custom_stencil3d_7pt_flat_f32_cpu(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6) {
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  for (int32_t i = 0; i < N; ++i) {
    float extra_v = extra ? extra[i] : 0.0f;
    float scale = coeff ? coeff[i] : 1.0f;
    float base = base0 * a0[i] + (extra ? base_extra * extra_v : 0.0f);
    float inner = c0 * a0[i] + c1 * a1[i] + c2 * a2[i] +
                  c3 * a3[i] + c4 * a4[i] + c5 * a5[i] +
                  c6 * a6[i] + (extra ? coeff_extra * extra_v : 0.0f);
    out[i] = base + scale * inner;
  }
  if (timing_enabled()) {
    fprintf(timing_file(),
            "POLYGEIST_RT_TIMING\top=customStencil3D7pt_f32_cpu_fallback"
            "\tm=%d\tn=1\tk=7\thost_ms=%.6f\tdevice_ms=0.000000\n",
            N, wall_time_ms() - host_start_ms);
    fflush(timing_file());
  }
}

void polygeist_custom_stencil3d_7pt_flat_f64(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6) {
  if (!polygeist_custom_stencil3d_7pt_flat_f64_device) {
    polygeist_custom_stencil3d_7pt_flat_f64_cpu(
        N, a0, a1, a2, a3, a4, a5, a6, extra, coeff, out,
        base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6);
    return;
  }
  if (N <= 0)
    return;

  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)N * sizeof(double);

  double *d0 = (double *)register_host_safe((void *)a0, bytes);
  double *d1 = (double *)register_host_safe((void *)a1, bytes);
  double *d2 = (double *)register_host_safe((void *)a2, bytes);
  double *d3 = (double *)register_host_safe((void *)a3, bytes);
  double *d4 = (double *)register_host_safe((void *)a4, bytes);
  double *d5 = (double *)register_host_safe((void *)a5, bytes);
  double *d6 = (double *)register_host_safe((void *)a6, bytes);
  double *dextra =
      extra ? (double *)register_host_safe((void *)extra, bytes) : NULL;
  double *dcoeff =
      coeff ? (double *)register_host_safe((void *)coeff, bytes) : NULL;
  double *dout = (double *)register_host_safe(out, bytes);

  timing_gpu_begin();
  polygeist_custom_stencil3d_7pt_flat_f64_device(
      N, d0, d1, d2, d3, d4, d5, d6, dextra, dcoeff, dout,
      base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6, g_stream);
  timing_gpu_end("customStencil3D7pt_f64", N, 1, 7, host_start_ms);

  unregister_host_safe((void *)a0);
  unregister_host_safe((void *)a1);
  unregister_host_safe((void *)a2);
  unregister_host_safe((void *)a3);
  unregister_host_safe((void *)a4);
  unregister_host_safe((void *)a5);
  unregister_host_safe((void *)a6);
  if (extra)
    unregister_host_safe((void *)extra);
  if (coeff)
    unregister_host_safe((void *)coeff);
  unregister_host_safe(out);
}

void polygeist_custom_stencil3d_7pt_flat_f32(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6) {
  if (!polygeist_custom_stencil3d_7pt_flat_f32_device) {
    polygeist_custom_stencil3d_7pt_flat_f32_cpu(
        N, a0, a1, a2, a3, a4, a5, a6, extra, coeff, out,
        base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6);
    return;
  }
  if (N <= 0)
    return;

  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)N * sizeof(float);

  float *d0 = (float *)register_host_safe((void *)a0, bytes);
  float *d1 = (float *)register_host_safe((void *)a1, bytes);
  float *d2 = (float *)register_host_safe((void *)a2, bytes);
  float *d3 = (float *)register_host_safe((void *)a3, bytes);
  float *d4 = (float *)register_host_safe((void *)a4, bytes);
  float *d5 = (float *)register_host_safe((void *)a5, bytes);
  float *d6 = (float *)register_host_safe((void *)a6, bytes);
  float *dextra =
      extra ? (float *)register_host_safe((void *)extra, bytes) : NULL;
  float *dcoeff =
      coeff ? (float *)register_host_safe((void *)coeff, bytes) : NULL;
  float *dout = (float *)register_host_safe(out, bytes);

  timing_gpu_begin();
  polygeist_custom_stencil3d_7pt_flat_f32_device(
      N, d0, d1, d2, d3, d4, d5, d6, dextra, dcoeff, dout,
      base0, base_extra, coeff_extra, c0, c1, c2, c3, c4, c5, c6, g_stream);
  timing_gpu_end("customStencil3D7pt_f32", N, 1, 7, host_start_ms);

  unregister_host_safe((void *)a0);
  unregister_host_safe((void *)a1);
  unregister_host_safe((void *)a2);
  unregister_host_safe((void *)a3);
  unregister_host_safe((void *)a4);
  unregister_host_safe((void *)a5);
  unregister_host_safe((void *)a6);
  if (extra)
    unregister_host_safe((void *)extra);
  if (coeff)
    unregister_host_safe((void *)coeff);
  unregister_host_safe(out);
}

static void polygeist_dft_z2z_1d_cpu(
    int32_t N, int32_t inverse, const double *A, double *B) {
  if (N <= 0) return;
  const double sign = inverse ? 1.0 : -1.0;
  for (int32_t k = 0; k < N; ++k) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int32_t n = 0; n < N; ++n) {
      double angle = sign * 2.0 * M_PI * (double)k * (double)n / (double)N;
      double c = cos(angle);
      double s = sin(angle);
      double ar = A[(size_t)2 * (size_t)n + 0];
      double ai = A[(size_t)2 * (size_t)n + 1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    B[(size_t)2 * (size_t)k + 0] = sum_re;
    B[(size_t)2 * (size_t)k + 1] = sum_im;
  }
}

static void polygeist_dft_c2c_1d_cpu(
    int32_t N, int32_t inverse, const float *A, float *B) {
  if (N <= 0) return;
  const float sign = inverse ? 1.0f : -1.0f;
  for (int32_t k = 0; k < N; ++k) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      float angle = sign * 2.0f * (float)M_PI * (float)k * (float)n / (float)N;
      float c = cosf(angle);
      float s = sinf(angle);
      float ar = A[(size_t)2 * (size_t)n + 0];
      float ai = A[(size_t)2 * (size_t)n + 1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    B[(size_t)2 * (size_t)k + 0] = sum_re;
    B[(size_t)2 * (size_t)k + 1] = sum_im;
  }
}

void polygeist_cufft_z2z_1d(
    int32_t N, int32_t inverse, const double *A, double *B) {
  if (N <= 0) return;
#if POLYGEIST_HAS_CUFFT
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  size_t bytes = (size_t)2 * (size_t)N * sizeof(double);
  cufftDoubleComplex *dA = NULL;
  cufftDoubleComplex *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes);
  DEVICE_MALLOC((void**)&dB, bytes);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes, cudaMemcpyHostToDevice, g_stream));

  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));
  CUFFT_CHECK(cufftSetStream(plan, g_stream));
  timing_gpu_begin();
  CUFFT_CHECK(cufftExecZ2Z(
      plan, dA, dB, inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
  timing_gpu_end("cufftZ2Z_1D", N, 1, inverse ? -1 : 1, host_start_ms);

  CUDA_CHECK(cudaMemcpyAsync(B, dB, bytes, cudaMemcpyDeviceToHost, g_stream));
  sync_stream_if_outside_pipeline();
  CUFFT_CHECK(cufftDestroy(plan));
  DEVICE_FREE(dA);
  DEVICE_FREE(dB);
#else
  polygeist_dft_z2z_1d_cpu(N, inverse, A, B);
#endif
}

void polygeist_cufft_c2c_1d(
    int32_t N, int32_t inverse, const float *A, float *B) {
  if (N <= 0) return;
#if POLYGEIST_HAS_CUFFT
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  size_t bytes = (size_t)2 * (size_t)N * sizeof(float);
  cufftComplex *dA = NULL;
  cufftComplex *dB = NULL;
  DEVICE_MALLOC((void**)&dA, bytes);
  DEVICE_MALLOC((void**)&dB, bytes);
  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes, cudaMemcpyHostToDevice, g_stream));

  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
  CUFFT_CHECK(cufftSetStream(plan, g_stream));
  timing_gpu_begin();
  CUFFT_CHECK(cufftExecC2C(
      plan, dA, dB, inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
  timing_gpu_end("cufftC2C_1D", N, 1, inverse ? -1 : 1, host_start_ms);

  CUDA_CHECK(cudaMemcpyAsync(B, dB, bytes, cudaMemcpyDeviceToHost, g_stream));
  sync_stream_if_outside_pipeline();
  CUFFT_CHECK(cufftDestroy(plan));
  DEVICE_FREE(dA);
  DEVICE_FREE(dB);
#else
  polygeist_dft_c2c_1d_cpu(N, inverse, A, B);
#endif
}

static void polygeist_cutensornet_tensor_product_3d_impl(
    int32_t KQ, int32_t KP, const void *psi, const void *u, void *out,
    size_t elementBytes, cudaDataType_t dataType) {
#if POLYGEIST_HAS_CUTENSORNET
  if (KQ <= 0 || KP <= 0) return;
  polygeist_cublas_init();

  size_t psiBytes = (size_t)KQ * (size_t)KP * elementBytes;
  size_t uBytes = (size_t)KP * (size_t)KP * (size_t)KP * elementBytes;
  size_t outBytes = (size_t)KQ * (size_t)KQ * (size_t)KQ * elementBytes;
  void *dPsi = register_host_safe((void *)psi, psiBytes);
  void *dU = register_host_safe((void *)u, uBytes);
  void *dOut = register_host_safe(out, outBytes);

  cutensornetHandle_t handle = NULL;
  cutensornetNetworkDescriptor_t network = NULL;
  cutensornetContractionOptimizerConfig_t config = NULL;
  cutensornetContractionOptimizerInfo_t info = NULL;
  cutensornetWorkspaceDescriptor_t workspace = NULL;
  void *scratch = NULL;

  // Row-major mode layouts for ai,bj,ck,ijk->abc.
  const int64_t psiExtents[2] = {KQ, KP};
  const int64_t uExtents[3] = {KP, KP, KP};
  const int64_t outExtents[3] = {KQ, KQ, KQ};
  const int64_t psiStrides[2] = {KP, 1};
  const int64_t uStrides[3] = {(int64_t)KP * KP, KP, 1};
  const int64_t outStrides[3] = {(int64_t)KQ * KQ, KQ, 1};
  const int32_t modesPsiA[2] = {'a', 'i'};
  const int32_t modesPsiB[2] = {'b', 'j'};
  const int32_t modesPsiC[2] = {'c', 'k'};
  const int32_t modesU[3] = {'i', 'j', 'k'};
  const int32_t modesOut[3] = {'a', 'b', 'c'};
  int64_t tensorIds[4] = {-1, -1, -1, -1};

  CUTENSORNET_CHECK(cutensornetCreate(&handle));
  CUTENSORNET_CHECK(cutensornetCreateNetwork(handle, &network));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      handle, network, 2, psiExtents, modesPsiA, NULL, dataType,
      &tensorIds[0]));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      handle, network, 2, psiExtents, modesPsiB, NULL, dataType,
      &tensorIds[1]));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      handle, network, 2, psiExtents, modesPsiC, NULL, dataType,
      &tensorIds[2]));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      handle, network, 3, uExtents, modesU, NULL, dataType,
      &tensorIds[3]));
  CUTENSORNET_CHECK(cutensornetNetworkSetOutputTensor(
      handle, network, 3, modesOut, dataType));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      handle, network, tensorIds[0], dPsi, psiStrides));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      handle, network, tensorIds[1], dPsi, psiStrides));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      handle, network, tensorIds[2], dPsi, psiStrides));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      handle, network, tensorIds[3], dU, uStrides));
  CUTENSORNET_CHECK(cutensornetNetworkSetOutputTensorMemory(
      handle, network, dOut, outStrides));
  CUTENSORNET_CHECK(
      cutensornetCreateContractionOptimizerConfig(handle, &config));
  CUTENSORNET_CHECK(
      cutensornetCreateContractionOptimizerInfo(handle, network, &info));
  const uint64_t workspaceLimit = UINT64_C(256) * 1024 * 1024;
  CUTENSORNET_CHECK(cutensornetContractionOptimize(
      handle, network, config, workspaceLimit, info));
  CUTENSORNET_CHECK(
      cutensornetNetworkSetOptimizerInfo(handle, network, info));
  CUTENSORNET_CHECK(cutensornetCreateWorkspaceDescriptor(handle, &workspace));
  CUTENSORNET_CHECK(cutensornetWorkspaceComputeContractionSizes(
      handle, network, info, workspace));
  int64_t scratchBytes = 0;
  CUTENSORNET_CHECK(cutensornetWorkspaceGetMemorySize(
      handle, workspace, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
      &scratchBytes));
  if (scratchBytes > 0)
    scratch = pipeline_device_malloc((size_t)scratchBytes);
  CUTENSORNET_CHECK(cutensornetWorkspaceSetMemory(
      handle, workspace, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, scratch, scratchBytes));
  CUTENSORNET_CHECK(
      cutensornetNetworkPrepareContraction(handle, network, workspace));
  CUTENSORNET_CHECK(cutensornetNetworkContract(
      handle, network, 0, workspace, NULL, g_stream));

  // The network objects are currently per-call, so contraction must finish
  // before their destruction. A future shape-keyed plan cache can remove
  // this synchronization and recover pipeline overlap.
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  pipeline_device_free(scratch);
  CUTENSORNET_CHECK(cutensornetDestroyWorkspaceDescriptor(workspace));
  CUTENSORNET_CHECK(cutensornetDestroyContractionOptimizerInfo(info));
  CUTENSORNET_CHECK(cutensornetDestroyContractionOptimizerConfig(config));
  CUTENSORNET_CHECK(cutensornetDestroyNetwork(network));
  CUTENSORNET_CHECK(cutensornetDestroy(handle));
  unregister_host_safe((void *)psi);
  unregister_host_safe((void *)u);
  unregister_host_safe(out);
#else
  (void)KQ; (void)KP; (void)psi; (void)u; (void)out;
  (void)elementBytes; (void)dataType;
  fprintf(stderr,
          "polygeist runtime: cuTensorNet support was not enabled at build "
          "time (define POLYGEIST_ENABLE_CUTENSORNET and link "
          "-lcutensornet -lcutensor)\n");
  abort();
#endif
}

void polygeist_cutensornet_tensor_product_3d_f32(
    int32_t KQ, int32_t KP, const float *psi, const float *u, float *out) {
  polygeist_cutensornet_tensor_product_3d_impl(
      KQ, KP, psi, u, out, sizeof(float), CUDA_R_32F);
}

void polygeist_cutensornet_tensor_product_3d_f64(
    int32_t KQ, int32_t KP, const double *psi, const double *u, double *out) {
  polygeist_cutensornet_tensor_product_3d_impl(
      KQ, KP, psi, u, out, sizeof(double), CUDA_R_64F);
}

enum { POLYGEIST_CONTRACTION_MAX_MODES = 64 };

static int polygeist_parse_contraction2_f64_metadata(
    const int64_t *metadata, int64_t ranks[3],
    int64_t extents[3][POLYGEIST_CONTRACTION_MAX_MODES],
    int64_t strides[3][POLYGEIST_CONTRACTION_MAX_MODES],
    int32_t modes[3][POLYGEIST_CONTRACTION_MAX_MODES],
    int present[3][POLYGEIST_CONTRACTION_MAX_MODES],
    int64_t modeExtents[POLYGEIST_CONTRACTION_MAX_MODES]) {
  enum {
    MAX_RANK = POLYGEIST_CONTRACTION_MAX_MODES,
    TENSOR_FIELDS = 3 * POLYGEIST_CONTRACTION_MAX_MODES
  };
  for (int mode = 0; mode < MAX_RANK; ++mode)
    modeExtents[mode] = 1;
  memset(present, 0, 3 * MAX_RANK * sizeof(int));

  for (int tensor = 0; tensor < 3; ++tensor) {
    ranks[tensor] = metadata[tensor];
    if (ranks[tensor] < 0 || ranks[tensor] > MAX_RANK)
      return 0;
    int64_t base = 3 + (int64_t)tensor * TENSOR_FIELDS;
    for (int64_t dim = 0; dim < ranks[tensor]; ++dim) {
      extents[tensor][dim] = metadata[base + dim];
      strides[tensor][dim] = metadata[base + MAX_RANK + dim];
      int64_t mode = metadata[base + 2 * MAX_RANK + dim];
      if (mode < 0 || mode >= MAX_RANK || extents[tensor][dim] <= 0 ||
          strides[tensor][dim] < 0)
        return 0;
      if (modeExtents[mode] != 1 &&
          modeExtents[mode] != extents[tensor][dim])
        return 0;
      modes[tensor][dim] = (int32_t)mode;
      modeExtents[mode] = extents[tensor][dim];
      present[tensor][mode] = 1;
    }
  }
  return 1;
}

#if POLYGEIST_HAS_CUTENSORNET

// A prepared cuTensorNet network is shape/layout specific but data-pointer
// independent: NetworkSet{Input,Output}TensorMemory may update the buffers
// before each contraction.  Keep the expensive optimizer and preparation
// result alive and only rebind pointers on a cache hit.
#define POLYGEIST_CONTRACTION_CACHE_CAP 64

typedef struct {
  int device;
  int64_t ranks[3];
  int64_t extents[3][POLYGEIST_CONTRACTION_MAX_MODES];
  int64_t strides[3][POLYGEIST_CONTRACTION_MAX_MODES];
  int32_t modes[3][POLYGEIST_CONTRACTION_MAX_MODES];
} PolygeistContractionKey;

typedef struct {
  int valid;
  uint64_t hash;
  uint64_t last_use;
  PolygeistContractionKey key;
  cutensornetNetworkDescriptor_t network;
  cutensornetContractionOptimizerConfig_t config;
  cutensornetContractionOptimizerInfo_t info;
  cutensornetWorkspaceDescriptor_t workspace;
  int64_t tensor_ids[2];
  void *scratch;
  int64_t scratch_bytes;
} PolygeistContractionCacheEntry;

static cutensornetHandle_t g_cutensornet_handle = NULL;
static PolygeistContractionCacheEntry
    g_contraction_cache[POLYGEIST_CONTRACTION_CACHE_CAP];
static uint64_t g_contraction_cache_clock = 0;
static uint64_t g_contraction_cache_hits = 0;
static uint64_t g_contraction_cache_misses = 0;
static uint64_t g_contraction_cache_evictions = 0;

static int contraction_cache_enabled(void) {
  const char *value = getenv("POLYGEIST_CUTENSORNET_PLAN_CACHE");
  return !value || !*value || strcmp(value, "0") != 0;
}

static int contraction_cache_stats_enabled(void) {
  const char *value = getenv("POLYGEIST_RT_CACHE_STATS");
  return value && *value && strcmp(value, "0") != 0;
}

static uint64_t hash_contraction_key(const PolygeistContractionKey *key) {
  const unsigned char *bytes = (const unsigned char *)key;
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < sizeof(*key); ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static void make_contraction_key(
    PolygeistContractionKey *key, const int64_t ranks[3],
    const int64_t extents[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int64_t strides[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int32_t modes[3][POLYGEIST_CONTRACTION_MAX_MODES]) {
  memset(key, 0, sizeof(*key));
  CUDA_CHECK(cudaGetDevice(&key->device));
  for (int tensor = 0; tensor < 3; ++tensor) {
    key->ranks[tensor] = ranks[tensor];
    for (int64_t dim = 0; dim < ranks[tensor]; ++dim) {
      key->extents[tensor][dim] = extents[tensor][dim];
      key->strides[tensor][dim] = strides[tensor][dim];
      key->modes[tensor][dim] = modes[tensor][dim];
    }
  }
}

static void ensure_cutensornet_handle(void) {
  if (!g_cutensornet_handle)
    CUTENSORNET_CHECK(cutensornetCreate(&g_cutensornet_handle));
}

static void destroy_contraction_cache_entry(
    PolygeistContractionCacheEntry *entry) {
  if (!entry->valid)
    return;
  if (entry->scratch)
    CUDA_CHECK(cudaFree(entry->scratch));
  if (entry->workspace)
    CUTENSORNET_CHECK(
        cutensornetDestroyWorkspaceDescriptor(entry->workspace));
  if (entry->info)
    CUTENSORNET_CHECK(
        cutensornetDestroyContractionOptimizerInfo(entry->info));
  if (entry->config)
    CUTENSORNET_CHECK(
        cutensornetDestroyContractionOptimizerConfig(entry->config));
  if (entry->network)
    CUTENSORNET_CHECK(cutensornetDestroyNetwork(entry->network));
  memset(entry, 0, sizeof(*entry));
}

static PolygeistContractionCacheEntry *create_contraction_cache_entry(
    PolygeistContractionCacheEntry *entry,
    const PolygeistContractionKey *key, uint64_t hash) {
  memset(entry, 0, sizeof(*entry));
  entry->hash = hash;
  entry->key = *key;
  entry->tensor_ids[0] = -1;
  entry->tensor_ids[1] = -1;
  ensure_cutensornet_handle();

  CUTENSORNET_CHECK(
      cutensornetCreateNetwork(g_cutensornet_handle, &entry->network));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      g_cutensornet_handle, entry->network, (int32_t)key->ranks[0],
      key->extents[0], key->modes[0], NULL, CUDA_R_64F,
      &entry->tensor_ids[0]));
  CUTENSORNET_CHECK(cutensornetNetworkAppendTensor(
      g_cutensornet_handle, entry->network, (int32_t)key->ranks[1],
      key->extents[1], key->modes[1], NULL, CUDA_R_64F,
      &entry->tensor_ids[1]));
  CUTENSORNET_CHECK(cutensornetNetworkSetOutputTensor(
      g_cutensornet_handle, entry->network, (int32_t)key->ranks[2],
      key->modes[2], CUDA_R_64F));

  CUTENSORNET_CHECK(cutensornetCreateContractionOptimizerConfig(
      g_cutensornet_handle, &entry->config));
  CUTENSORNET_CHECK(cutensornetCreateContractionOptimizerInfo(
      g_cutensornet_handle, entry->network, &entry->info));
  const uint64_t workspace_limit = UINT64_C(256) * 1024 * 1024;
  CUTENSORNET_CHECK(cutensornetContractionOptimize(
      g_cutensornet_handle, entry->network, entry->config,
      workspace_limit, entry->info));
  CUTENSORNET_CHECK(cutensornetNetworkSetOptimizerInfo(
      g_cutensornet_handle, entry->network, entry->info));
  CUTENSORNET_CHECK(cutensornetCreateWorkspaceDescriptor(
      g_cutensornet_handle, &entry->workspace));
  CUTENSORNET_CHECK(cutensornetWorkspaceComputeContractionSizes(
      g_cutensornet_handle, entry->network, entry->info, entry->workspace));
  CUTENSORNET_CHECK(cutensornetWorkspaceGetMemorySize(
      g_cutensornet_handle, entry->workspace,
      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, &entry->scratch_bytes));
  if (entry->scratch_bytes > 0)
    CUDA_CHECK(cudaMalloc(&entry->scratch, (size_t)entry->scratch_bytes));
  CUTENSORNET_CHECK(cutensornetWorkspaceSetMemory(
      g_cutensornet_handle, entry->workspace, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, entry->scratch, entry->scratch_bytes));
  CUTENSORNET_CHECK(cutensornetNetworkPrepareContraction(
      g_cutensornet_handle, entry->network, entry->workspace));
  entry->valid = 1;
  entry->last_use = ++g_contraction_cache_clock;
  return entry;
}

static PolygeistContractionCacheEntry *get_contraction_cache_entry(
    const PolygeistContractionKey *key) {
  const uint64_t hash = hash_contraction_key(key);
  for (int i = 0; i < POLYGEIST_CONTRACTION_CACHE_CAP; ++i) {
    PolygeistContractionCacheEntry *entry = &g_contraction_cache[i];
    if (entry->valid && entry->hash == hash &&
        memcmp(&entry->key, key, sizeof(*key)) == 0) {
      g_contraction_cache_hits++;
      entry->last_use = ++g_contraction_cache_clock;
      return entry;
    }
  }

  g_contraction_cache_misses++;
  int slot = -1;
  for (int i = 0; i < POLYGEIST_CONTRACTION_CACHE_CAP; ++i) {
    if (!g_contraction_cache[i].valid) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    slot = 0;
    for (int i = 1; i < POLYGEIST_CONTRACTION_CACHE_CAP; ++i)
      if (g_contraction_cache[i].last_use <
          g_contraction_cache[slot].last_use)
        slot = i;
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    destroy_contraction_cache_entry(&g_contraction_cache[slot]);
    g_contraction_cache_evictions++;
  }
  return create_contraction_cache_entry(&g_contraction_cache[slot], key,
                                        hash);
}

static void destroy_cutensornet_contraction_cache(void) {
  for (int i = 0; i < POLYGEIST_CONTRACTION_CACHE_CAP; ++i)
    destroy_contraction_cache_entry(&g_contraction_cache[i]);
  if (g_cutensornet_handle) {
    CUTENSORNET_CHECK(cutensornetDestroy(g_cutensornet_handle));
    g_cutensornet_handle = NULL;
  }
  if (contraction_cache_stats_enabled()) {
    fprintf(stderr,
            "POLYGEIST_RT_CACHE_STATS\thits=%llu\tmisses=%llu\t"
            "evictions=%llu\n",
            (unsigned long long)g_contraction_cache_hits,
            (unsigned long long)g_contraction_cache_misses,
            (unsigned long long)g_contraction_cache_evictions);
  }
  memset(g_contraction_cache, 0, sizeof(g_contraction_cache));
  g_contraction_cache_clock = 0;
  g_contraction_cache_hits = 0;
  g_contraction_cache_misses = 0;
  g_contraction_cache_evictions = 0;
}

#endif // POLYGEIST_HAS_CUTENSORNET

static void polygeist_contraction2_f64_cpu(
    const double *A, const double *B, double *C,
    const int64_t ranks[3],
    const int64_t extents[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int64_t strides[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int32_t modes[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int present[3][POLYGEIST_CONTRACTION_MAX_MODES],
    const int64_t modeExtents[POLYGEIST_CONTRACTION_MAX_MODES]) {
  int64_t total = 1;
  for (int mode = 0; mode < POLYGEIST_CONTRACTION_MAX_MODES; ++mode)
    total *= modeExtents[mode];
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t coordinates[POLYGEIST_CONTRACTION_MAX_MODES];
    int64_t remaining = linear;
    for (int mode = POLYGEIST_CONTRACTION_MAX_MODES - 1;
         mode >= 0; --mode) {
      coordinates[mode] = remaining % modeExtents[mode];
      remaining /= modeExtents[mode];
    }
    int64_t offsets[3] = {0, 0, 0};
    for (int tensor = 0; tensor < 3; ++tensor)
      for (int64_t dim = 0; dim < ranks[tensor]; ++dim)
        offsets[tensor] +=
            coordinates[modes[tensor][dim]] * strides[tensor][dim];
    int firstReductionPoint = 1;
    for (int mode = 0; mode < POLYGEIST_CONTRACTION_MAX_MODES; ++mode)
      if (!present[2][mode] &&
          (present[0][mode] || present[1][mode]) &&
          coordinates[mode] != 0)
        firstReductionPoint = 0;
    if (firstReductionPoint)
      C[offsets[2]] = 0.0;
    C[offsets[2]] += A[offsets[0]] * B[offsets[1]];
  }
}

static void polygeist_cutensornet_contraction2_f64_impl(
    const double *A, const double *B, double *C, const int64_t *metadata,
    int device_pointers) {
  int64_t ranks[3];
  int64_t extents[3][POLYGEIST_CONTRACTION_MAX_MODES] = {{0}};
  int64_t strides[3][POLYGEIST_CONTRACTION_MAX_MODES] = {{0}};
  int32_t modes[3][POLYGEIST_CONTRACTION_MAX_MODES] = {{0}};
  int present[3][POLYGEIST_CONTRACTION_MAX_MODES];
  int64_t modeExtents[POLYGEIST_CONTRACTION_MAX_MODES];
  if (!polygeist_parse_contraction2_f64_metadata(
          metadata, ranks, extents, strides, modes, present, modeExtents)) {
    fprintf(stderr, "polygeist runtime: invalid contraction metadata\n");
    abort();
  }

#if POLYGEIST_HAS_CUTENSORNET
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  polygeist_cublas_init();
  size_t elements[3] = {1, 1, 1};
  for (int tensor = 0; tensor < 3; ++tensor)
    for (int64_t dim = 0; dim < ranks[tensor]; ++dim)
      elements[tensor] +=
          (size_t)(extents[tensor][dim] - 1) *
          (size_t)strides[tensor][dim];
  double *dA = device_pointers
                    ? (double *)A
                    : (double *)register_host_safe(
                          (void *)A, elements[0] * sizeof(double));
  double *dB = device_pointers
                    ? (double *)B
                    : (double *)register_host_safe(
                          (void *)B, elements[1] * sizeof(double));
  double *dC = device_pointers
                    ? C
                    : (double *)register_host_safe(
                          (void *)C, elements[2] * sizeof(double));

  PolygeistContractionKey key;
  make_contraction_key(&key, ranks, extents, strides, modes);
  PolygeistContractionCacheEntry uncached_entry;
  const int use_cache = contraction_cache_enabled();
  PolygeistContractionCacheEntry *entry =
      use_cache ? get_contraction_cache_entry(&key)
                : create_contraction_cache_entry(
                      &uncached_entry, &key, hash_contraction_key(&key));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      g_cutensornet_handle, entry->network, entry->tensor_ids[0], dA,
      strides[0]));
  CUTENSORNET_CHECK(cutensornetNetworkSetInputTensorMemory(
      g_cutensornet_handle, entry->network, entry->tensor_ids[1], dB,
      strides[1]));
  CUTENSORNET_CHECK(cutensornetNetworkSetOutputTensorMemory(
      g_cutensornet_handle, entry->network, dC, strides[2]));
  timing_gpu_begin();
  CUTENSORNET_CHECK(cutensornetNetworkContract(
      g_cutensornet_handle, entry->network, 0, entry->workspace, NULL,
      g_stream));
  int64_t reductionExtent = 1;
  for (int mode = 0; mode < POLYGEIST_CONTRACTION_MAX_MODES; ++mode)
    if (!present[2][mode] &&
        (present[0][mode] || present[1][mode]))
      reductionExtent *= modeExtents[mode];
  timing_gpu_end("cutensornetContraction2_f64",
                 (int32_t)modeExtents[0], (int32_t)modeExtents[1],
                 (int32_t)reductionExtent, host_start_ms);
  if (use_cache) {
    sync_stream_if_outside_pipeline();
  } else {
    // The uncached debug path owns descriptors and scratch only for this
    // invocation, so execution must finish before they are destroyed.
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    destroy_contraction_cache_entry(entry);
  }
  if (!device_pointers) {
    unregister_host_safe((void *)A);
    unregister_host_safe((void *)B);
    unregister_host_safe(C);
  }
#else
  polygeist_contraction2_f64_cpu(
      A, B, C, ranks, extents, strides, modes, present, modeExtents);
#endif
}

void polygeist_cutensornet_contraction2_f64(
    const double *A, const double *B, double *C, const int64_t *metadata) {
  polygeist_cutensornet_contraction2_f64_impl(A, B, C, metadata, 0);
}

void polygeist_cutensornet_contraction2_f64_device(
    const double *A, const double *B, double *C, const int64_t *metadata) {
  polygeist_cutensornet_contraction2_f64_impl(A, B, C, metadata, 1);
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
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

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
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
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
  DEVICE_MALLOC((void**)&dA, bytes_in);
  DEVICE_MALLOC((void**)&dF, bytes_f);
  DEVICE_MALLOC((void**)&dB, bytes_out);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

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
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dA);  DEVICE_FREE(dF);  DEVICE_FREE(dB);
  if (dWS) DEVICE_FREE(dWS);
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
  pipeline_host_free(A32);
  pipeline_host_free(B32);
}

// ============================================================================
// Extracted-darknet batched CNN-block primitives. All FP32, NCHW.
//
// MEMORY MODEL: same zero-copy pattern as the BLAS shims —
// cudaHostRegister + cudaHostGetDevicePointer via register_host_safe().
// On Jetson Orin's iGPU these calls just set up the page-table mapping
// (no bytes move). Workspace allocations route through the pipeline temp
// cache when `polygeist_cublas_pipeline_begin/end` scopes are active.
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dO));
  sync_stream_if_outside_pipeline();

  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
}

void polygeist_cudnn_conv3d_channels_f32(
    int32_t IC, int32_t inD, int32_t inH, int32_t inW,
    int32_t OC, int32_t kD, int32_t kH, int32_t kW,
    const float *input, const float *filter, const float *bias, float *output) {
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  const int32_t outD = inD - kD + 1;
  const int32_t outH = inH - kH + 1;
  const int32_t outW = inW - kW + 1;

  size_t inputBytes = (size_t)IC * inD * inH * inW * sizeof(float);
  size_t filterBytes =
      (size_t)OC * IC * kD * kH * kW * sizeof(float);
  size_t outputBytes =
      (size_t)OC * outD * outH * outW * sizeof(float);
  float *dInput = (float *)register_host_safe((void *)input, inputBytes);
  float *dFilter = (float *)register_host_safe((void *)filter, filterBytes);
  float *dOutput = (float *)register_host_safe(output, outputBytes);
  float *dBias = bias ? (float *)register_host_safe(
                           (void *)bias, (size_t)OC * sizeof(float))
                      : NULL;

  cudnnTensorDescriptor_t inputDesc, outputDesc, biasDesc = NULL;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
  int inputDims[5] = {1, IC, inD, inH, inW};
  int inputStrides[5] = {IC * inD * inH * inW, inD * inH * inW,
                         inH * inW, inW, 1};
  int outputDims[5] = {1, OC, outD, outH, outW};
  int outputStrides[5] = {OC * outD * outH * outW, outD * outH * outW,
                          outH * outW, outW, 1};
  int filterDims[5] = {OC, IC, kD, kH, kW};
  int pad[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  int dilation[3] = {1, 1, 1};
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      inputDesc, CUDNN_DATA_FLOAT, 5, inputDims, inputStrides));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDims));
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      convDesc, 3, pad, stride, dilation,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      outputDesc, CUDNN_DATA_FLOAT, 5, outputDims, outputStrides));

  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int returned = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      g_cudnn, inputDesc, filterDesc, convDesc, outputDesc,
      1, &returned, &algoPerf));
  if (returned < 1) {
    fprintf(stderr, "cuDNN channel Conv3D: no forward algorithm available\n");
    abort();
  }
  size_t workspaceBytes = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      g_cudnn, inputDesc, filterDesc, convDesc, outputDesc,
      algoPerf.algo, &workspaceBytes));
  void *workspace = NULL;
  if (workspaceBytes) DEVICE_MALLOC(&workspace, workspaceBytes);

  float one = 1.0f, zero = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &one, inputDesc, dInput, filterDesc, dFilter, convDesc,
      algoPerf.algo, workspace, workspaceBytes, &zero, outputDesc, dOutput));
  if (dBias) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
    int biasDims[5] = {1, OC, 1, 1, 1};
    int biasStrides[5] = {OC, 1, 1, 1, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        biasDesc, CUDNN_DATA_FLOAT, 5, biasDims, biasStrides));
    CUDNN_CHECK(cudnnAddTensor(
        g_cudnn, &one, biasDesc, dBias, &one, outputDesc, dOutput));
  }
  timing_gpu_end("cudnnConvolution3D_channels_f32",
                 OC * outD, outH * outW, IC * kD * kH * kW,
                 host_start_ms);
  sync_stream_if_outside_pipeline();

  if (workspace) DEVICE_FREE(workspace);
  if (biasDesc) cudnnDestroyTensorDescriptor(biasDesc);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
  unregister_host_safe((void *)input);
  unregister_host_safe((void *)filter);
  if (bias) unregister_host_safe((void *)bias);
  unregister_host_safe(output);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  float alpha = 1.0f, beta = 0.0f;
  timing_gpu_begin();
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dO));
  timing_gpu_end("cudnnConv2d_im2col_gemm", OC, OH * OW, IC * K * K,
                 host_start_ms);

  if (dWS) DEVICE_FREE(dWS);
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
  sync_stream_if_outside_pipeline();

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
  if (!var_h) {
    fprintf(stderr, "polygeist_cudnn_batchnorm_inference: malloc failed\n");
    abort();
  }
  const float *inv_std_h = inv_std;
  void *inv_std_device = NULL;
  if (pointer_is_device_resident(inv_std, &inv_std_device)) {
    CUDA_CHECK(cudaMemcpy(var_h, inv_std_device, (size_t)C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    inv_std_h = var_h;
  }
  for (int32_t c = 0; c < C; ++c) {
    double s = (double)inv_std_h[c];
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
  DEVICE_MALLOC((void **)&dV, bytes_c);
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
  sync_stream_if_outside_pipeline();

  DEVICE_FREE(dV);
  pipeline_host_free(var_h);
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
  sync_stream_if_outside_pipeline();

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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  // y = relu(1·conv(A, F) + 1·Z + bias).
  float alpha1 = 1.0f, alpha2 = 1.0f;
  CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
      g_cudnn, &alpha1, in_desc, dA, f_desc, dF, conv_desc, algo,
      dWS, ws_size, &alpha2, out_desc, dZ,
      bias_desc, dB, act_desc, out_desc, dO));
  sync_stream_if_outside_pipeline();

  if (dWS) DEVICE_FREE(dWS);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(f_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyActivationDescriptor(act_desc);
}

void polygeist_cublas_memset_zero_2d_f32(int32_t M, int32_t N, float *A, int32_t lda) {
  void *device_ptr = NULL;
  if (pointer_is_device_resident(A, &device_ptr)) {
    polygeist_cublas_init();
    double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
    timing_gpu_begin();
    CUDA_CHECK(cudaMemset2DAsync(device_ptr, (size_t)lda * sizeof(float), 0,
                                 (size_t)N * sizeof(float), M, g_stream));
    timing_gpu_end("cuda_memset_zero_2d_f32", M, N, 0, host_start_ms);
    return;
  }
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
  sync_stream_if_outside_pipeline();
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
  sync_stream_if_outside_pipeline();

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
  if (heur.workspaceSize > 0) DEVICE_MALLOC(&dWS, heur.workspaceSize);

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
  sync_stream_if_outside_pipeline();

  if (dWS) DEVICE_FREE(dWS);
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
  DEVICE_MALLOC((void **)&dF, bytes_F);
  DEVICE_MALLOC((void **)&dB, bytes_b);
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
  if (ws_size > 0) DEVICE_MALLOC(&dWS, ws_size);

  // y = act(α₁ * conv(x, w') + α₂ * z + b'). We want α₂ = 0 so z is
  // unused — but cuDNN requires a valid z descriptor + pointer anyway.
  // Reuse the output buffer as z (cuDNN accepts that when α₂ = 0).
  float alpha1 = 1.0f, alpha2 = 0.0f;
  CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
      g_cudnn, &alpha1, in_desc, dA, f_desc, dF, conv_desc, algo,
      dWS, ws_size, &alpha2, out_desc, dO,
      bias_desc, dB, act_desc, out_desc, dO));
  sync_stream_if_outside_pipeline();

  if (dWS) DEVICE_FREE(dWS);
  DEVICE_FREE(dF);
  DEVICE_FREE(dB);
  pipeline_host_free(F_fold);
  pipeline_host_free(b_fold);
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
    DEVICE_FREE(p->workspace);
    p->workspace = NULL;
  }
  if (p->dBias) {
    DEVICE_FREE(p->dBias);
    p->dBias = NULL;
  }
  if (p->dOut) {
    DEVICE_FREE(p->dOut);
    p->dOut = NULL;
  }
  if (p->dWeight) {
    DEVICE_FREE(p->dWeight);
    p->dWeight = NULL;
  }
  if (p->dX) {
    DEVICE_FREE(p->dX);
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
  DEVICE_MALLOC((void **)&p->dX, p->bytes);
  DEVICE_MALLOC((void **)&p->dWeight, p->bytes);
  DEVICE_MALLOC((void **)&p->dOut, p->bytes);
  DEVICE_MALLOC((void **)&p->dBias, p->bytes);
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
    DEVICE_MALLOC(&p->workspace, (size_t)workspace_size);

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

  sync_stream_if_outside_pipeline();
  rmsnorm_host_f32(N, X, Weight, Out);

  timing_host_only("host_rmsnorm_f32", N, 1, 0, host_start_ms);
}

void polygeist_rmsnorm_unweighted_f32(
    int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);

  float ss = 0.0f;
  timing_gpu_begin();
  CUBLAS_CHECK(cublasSdot(g_handle, N, dX, 1, dX, 1, &ss));
  CUDA_CHECK(cudaMemcpyAsync(dOut, dX, bytes, cudaMemcpyDeviceToDevice,
                             g_stream));
  float scale = 1.0f / sqrtf(ss / (float)N + 1.0e-5f);
  CUBLAS_CHECK(cublasSscal(g_handle, N, &scale, dOut, 1));
  timing_gpu_end("cublasRmsNormUnweighted_f32", N, 1, 0, host_start_ms);
}

void polygeist_cublas_dot_f32(
    int32_t N, const float *X, const float *Y, float *Out) {
  if (N <= 0) {
    void *device_out = NULL;
    if (pointer_is_device_resident(Out, &device_out))
      CUDA_CHECK(cudaMemset(device_out, 0, sizeof(float)));
    else
      *Out = 0.0f;
    return;
  }
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dY = (float *)register_host_safe((void *)Y, bytes);
  void *device_out = NULL;
  int out_is_device = pointer_is_device_resident(Out, &device_out);
  float host_out = 0.0f;

  timing_gpu_begin();
  CUBLAS_CHECK(cublasSdot(g_handle, N, dX, 1, dY, 1,
                          out_is_device ? &host_out : Out));
  if (out_is_device)
    CUDA_CHECK(cudaMemcpy(device_out, &host_out, sizeof(float),
                          cudaMemcpyHostToDevice));
  timing_gpu_end("cublasSdot", N, 1, 0, host_start_ms);
}

void polygeist_cublas_dot_f64(
    int32_t N, const double *X, const double *Y, double *Out) {
  if (N <= 0) {
    void *device_out = NULL;
    if (pointer_is_device_resident(Out, &device_out))
      CUDA_CHECK(cudaMemset(device_out, 0, sizeof(double)));
    else
      *Out = 0.0;
    return;
  }
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(double);
  double *dX = (double *)register_host_safe((void *)X, bytes);
  double *dY = (double *)register_host_safe((void *)Y, bytes);
  void *device_out = NULL;
  int out_is_device = pointer_is_device_resident(Out, &device_out);
  double host_out = 0.0;

  timing_gpu_begin();
  CUBLAS_CHECK(cublasDdot(g_handle, N, dX, 1, dY, 1,
                          out_is_device ? &host_out : Out));
  if (out_is_device)
    CUDA_CHECK(cudaMemcpy(device_out, &host_out, sizeof(double),
                          cudaMemcpyHostToDevice));
  timing_gpu_end("cublasDdot", N, 1, 0, host_start_ms);
}

void polygeist_cuda_gelu_tanh_f32(
    int32_t N, const float *X, float *Out) {
  if (N <= 0) return;
  polygeist_cublas_init();
  ensure_cudnn();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;

  size_t bytes = (size_t)N * sizeof(float);
  float *dX = (float *)register_host_safe((void *)X, bytes);
  float *dOut = (float *)register_host_safe(Out, bytes);
  float *dTmp0 = NULL;
  float *dTmp1 = NULL;
  float *dOnes = NULL;
  float *ones = (float *)malloc(bytes);
  if (!ones) {
    fprintf(stderr, "polygeist_cuda_gelu_tanh_f32: malloc failed\n");
    abort();
  }
  for (int32_t i = 0; i < N; ++i)
    ones[i] = 1.0f;
  DEVICE_MALLOC((void **)&dTmp0, bytes);
  DEVICE_MALLOC((void **)&dTmp1, bytes);
  DEVICE_MALLOC((void **)&dOnes, bytes);

  cudnnTensorDescriptor_t desc;
  cudnnActivationDescriptor_t tanh_desc;
  cudnnOpTensorDescriptor_t mul_desc, add_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, 1, 1, N));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&tanh_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      tanh_desc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_desc));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      mul_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&add_desc));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(
      add_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

  const float one = 1.0f;
  const float zero = 0.0f;
  const float half = 0.5f;
  const float cube_coeff = 0.044715f;
  const float sqrt_2_over_pi = 0.7978845608028654f;

  timing_gpu_begin();
  CUDA_CHECK(cudaMemcpyAsync(dOnes, ones, bytes, cudaMemcpyHostToDevice,
                             g_stream));
  // tmp0 = x * x
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, desc, dX,
                            &one, desc, dX,
                            &zero, desc, dTmp0));
  // tmp0 = tmp0 * x = x^3
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &one, desc, dTmp0,
                            &one, desc, dX,
                            &zero, desc, dTmp0));
  // tmp1 = x + 0.044715 * x^3
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, add_desc,
                            &one, desc, dX,
                            &cube_coeff, desc, dTmp0,
                            &zero, desc, dTmp1));
  CUBLAS_CHECK(cublasSscal(g_handle, N, &sqrt_2_over_pi, dTmp1, 1));
  CUDNN_CHECK(cudnnActivationForward(g_cudnn, tanh_desc,
                                     &one, desc, dTmp1,
                                     &zero, desc, dTmp1));
  // tmp1 = 1 + tanh(...)
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, add_desc,
                            &one, desc, dTmp1,
                            &one, desc, dOnes,
                            &zero, desc, dTmp1));
  // out = 0.5 * x * tmp1
  CUDNN_CHECK(cudnnOpTensor(g_cudnn, mul_desc,
                            &half, desc, dX,
                            &one, desc, dTmp1,
                            &zero, desc, dOut));
  timing_gpu_end("cudaGeluTanh_f32", N, 1, 0, host_start_ms);

  cudnnDestroyOpTensorDescriptor(mul_desc);
  cudnnDestroyOpTensorDescriptor(add_desc);
  cudnnDestroyActivationDescriptor(tanh_desc);
  cudnnDestroyTensorDescriptor(desc);
  DEVICE_FREE(dTmp0);
  DEVICE_FREE(dTmp1);
  DEVICE_FREE(dOnes);
  pipeline_host_free(ones);
}

void polygeist_whisper_exp_shift_sum_f32(
    int32_t N, const float *X, float max_val, float *Out, float *Sum) {
  if (N <= 0) {
    *Sum = 0.0f;
    return;
  }
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  float sum = 0.0f;
  for (int32_t i = 0; i < N; ++i) {
    float v = expf(X[i] - max_val);
    Out[i] = v;
    sum += v;
  }
  *Sum = sum;
  timing_host_only("hostWhisperExpShiftSum_f32", N, 1, 0, host_start_ms);
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

void polygeist_cuda_copy_strided_2d_f32(
    int32_t rows, int32_t cols,
    int32_t src_row_stride, int32_t src_col_stride,
    int32_t dst_row_stride, int32_t dst_col_stride,
    const float *X, float *Out) {
  if (rows <= 0 || cols <= 0) return;
  if (src_col_stride != 1 || dst_col_stride != 1) {
    fprintf(stderr, "cudaCopy strided f32 requires unit inner strides\n");
    abort();
  }
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t src_elems = (size_t)(rows - 1) * (size_t)src_row_stride + cols;
  size_t dst_elems = (size_t)(rows - 1) * (size_t)dst_row_stride + cols;
  float *dX = (float *)register_host_safe((void *)X, src_elems * sizeof(float));
  float *dOut = (float *)register_host_safe(Out, dst_elems * sizeof(float));
  timing_gpu_begin();
  if (src_row_stride == cols && dst_row_stride == cols) {
    CUDA_CHECK(cudaMemcpyAsync(dOut, dX, (size_t)rows * cols * sizeof(float),
                               cudaMemcpyDeviceToDevice, g_stream));
  } else {
    CUDA_CHECK(cudaMemcpy2DAsync(
        dOut, (size_t)dst_row_stride * sizeof(float),
        dX, (size_t)src_row_stride * sizeof(float),
        (size_t)cols * sizeof(float), rows,
        cudaMemcpyDeviceToDevice, g_stream));
  }
  timing_gpu_end("cudaCopyStrided2D_f32", rows, cols, 0, host_start_ms);
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
  DEVICE_MALLOC((void **)&dKeep, bytes);
  DEVICE_MALLOC((void **)&dBias, bytes);

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
  DEVICE_FREE(dKeep);
  DEVICE_FREE(dBias);
  pipeline_host_free(keep_h);
  pipeline_host_free(bias_h);
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
  DEVICE_MALLOC((void **)&dSigmoid, bytes);

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
  DEVICE_FREE(dSigmoid);
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
  DEVICE_MALLOC((void **)&dTmp, mat_bytes);

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
  DEVICE_FREE(dTmp);
}

#if POLYGEIST_HAS_CUTENSOR
static cutensorOperator_t cutensor_unary_operator(int32_t op) {
  switch (op) {
  case POLYGEIST_CUTENSOR_UNARY_ABS: return CUTENSOR_OP_ABS;
  case POLYGEIST_CUTENSOR_UNARY_ACOS: return CUTENSOR_OP_ACOS;
  case POLYGEIST_CUTENSOR_UNARY_ACOSH: return CUTENSOR_OP_ACOSH;
  case POLYGEIST_CUTENSOR_UNARY_ASIN: return CUTENSOR_OP_ASIN;
  case POLYGEIST_CUTENSOR_UNARY_ASINH: return CUTENSOR_OP_ASINH;
  case POLYGEIST_CUTENSOR_UNARY_ATAN: return CUTENSOR_OP_ATAN;
  case POLYGEIST_CUTENSOR_UNARY_ATANH: return CUTENSOR_OP_ATANH;
  case POLYGEIST_CUTENSOR_UNARY_CEIL: return CUTENSOR_OP_CEIL;
  case POLYGEIST_CUTENSOR_UNARY_COS: return CUTENSOR_OP_COS;
  case POLYGEIST_CUTENSOR_UNARY_COSH: return CUTENSOR_OP_COSH;
  case POLYGEIST_CUTENSOR_UNARY_EXP: return CUTENSOR_OP_EXP;
  case POLYGEIST_CUTENSOR_UNARY_FLOOR: return CUTENSOR_OP_FLOOR;
  case POLYGEIST_CUTENSOR_UNARY_LOG: return CUTENSOR_OP_LOG;
  case POLYGEIST_CUTENSOR_UNARY_MISH: return CUTENSOR_OP_MISH;
  case POLYGEIST_CUTENSOR_UNARY_NEG: return CUTENSOR_OP_NEG;
  case POLYGEIST_CUTENSOR_UNARY_RECIPROCAL: return CUTENSOR_OP_RCP;
  case POLYGEIST_CUTENSOR_UNARY_RELU: return CUTENSOR_OP_RELU;
  case POLYGEIST_CUTENSOR_UNARY_SIGMOID: return CUTENSOR_OP_SIGMOID;
  case POLYGEIST_CUTENSOR_UNARY_SILU: return CUTENSOR_OP_SWISH;
  case POLYGEIST_CUTENSOR_UNARY_SIN: return CUTENSOR_OP_SIN;
  case POLYGEIST_CUTENSOR_UNARY_SINH: return CUTENSOR_OP_SINH;
  case POLYGEIST_CUTENSOR_UNARY_SQRT: return CUTENSOR_OP_SQRT;
  case POLYGEIST_CUTENSOR_UNARY_TAN: return CUTENSOR_OP_TAN;
  case POLYGEIST_CUTENSOR_UNARY_TANH: return CUTENSOR_OP_TANH;
  default:
    fprintf(stderr, "polygeist_cutensor_unary_f32: invalid op %d\n", op);
    abort();
  }
}
#endif

void polygeist_cutensor_unary_f32(
    int32_t op, int32_t n, const float *x, float *out) {
  if (n <= 0) return;
#if POLYGEIST_HAS_CUTENSOR
  polygeist_cublas_init();
  double host_start_ms = timing_enabled() ? wall_time_ms() : 0.0;
  size_t bytes = (size_t)n * sizeof(float);
  float *dx = (float *)register_host_safe((void *)x, bytes);
  float *dout = (float *)register_host_safe(out, bytes);
  int64_t extent[1] = {n};
  int64_t stride[1] = {1};
  int32_t mode[1] = {0};
  float alpha = 1.0f;
  cutensorHandle_t handle = NULL;
  cutensorTensorDescriptor_t desc_x = NULL, desc_out = NULL;
  cutensorOperationDescriptor_t operation = NULL;
  cutensorPlanPreference_t preference = NULL;
  cutensorPlan_t plan = NULL;
  CUTENSOR_CHECK(cutensorCreate(&handle));
  CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
      handle, &desc_x, 1, extent, stride, CUDA_R_32F, 128));
  CUTENSOR_CHECK(cutensorCreateTensorDescriptor(
      handle, &desc_out, 1, extent, stride, CUDA_R_32F, 128));
  CUTENSOR_CHECK(cutensorCreatePermutation(
      handle, &operation, desc_x, mode, cutensor_unary_operator(op),
      desc_out, mode, CUTENSOR_COMPUTE_DESC_32F));
  CUTENSOR_CHECK(cutensorCreatePlanPreference(
      handle, &preference, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
  CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, operation, preference, 0));
  timing_gpu_begin();
  CUTENSOR_CHECK(cutensorPermute(handle, plan, &alpha, dx, dout, g_stream));
  timing_gpu_end("cutensorUnary_f32", n, op, 0, host_start_ms);
  CUTENSOR_CHECK(cutensorDestroyPlan(plan));
  CUTENSOR_CHECK(cutensorDestroyPlanPreference(preference));
  CUTENSOR_CHECK(cutensorDestroyOperationDescriptor(operation));
  CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(desc_x));
  CUTENSOR_CHECK(cutensorDestroyTensorDescriptor(desc_out));
  CUTENSOR_CHECK(cutensorDestroy(handle));
#else
  (void)op; (void)x; (void)out;
  fprintf(stderr,
          "polygeist_cutensor_unary_f32 requires "
          "-DPOLYGEIST_ENABLE_CUTENSOR and -lcutensor\n");
  abort();
#endif
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
