/*===- polygeist_v2_rt_cuda.c - lib_v2 ABI on cuBLAS / cuSOLVER -----------===*
 *
 * CUDA implementation of polygeist_v2_rt.h: thin pass-through wrappers around
 * the cuBLAS (v2 API) and cuSOLVER-Dn routines named by the lib_v2 manifests.
 *
 * Device data handling (deliberately simple, correct, not fast):
 *   * Every call is self-contained.  The host footprints of all array
 *     operands are merged into disjoint intervals (so operands that alias in
 *     host memory -- e.g. an in-place trmm with C == B -- alias on the device
 *     too), each interval is copied host->device with cudaMemcpy, the library
 *     routine runs on the legacy default stream, and every interval that
 *     contains a written operand is copied back device->host before return.
 *     Untouched gaps inside an interval (rows beyond the view when ld > m)
 *     round-trip unchanged.
 *   * Device buffers are grow-only per-slot scratch allocations, reused across
 *     calls, so steady-state calls do no cudaMalloc.
 *   * cuBLAS uses host pointer mode: alpha/beta/result live on the host.
 *   * The CUDA context, cuBLAS handle and cuSOLVER handle are created in a
 *     constructor so the one-off initialisation is outside timed regions.
 *
 * Performance caveat: on Jetson Orin the GPU shares DRAM with the CPU, so the
 * staged copies are pure overhead (pageable memcpy, no residency across
 * calls).  Keeping arrays resident across consecutive calls (as the
 * production runtime's static residency helpers in polygeist_cublas_rt_cuda.c
 * do) or zero-copy mapping would remove most of it; sharing that machinery
 * needs a decision by the runtime's owner, so it is not reused here.
 *
 * Library-ABI conventions handled generically by the generated code:
 *   cuSOLVER legacy API: `<call>_bufferSize` workspace query, device
 *   workspace, device status word (`devInfo`) checked after the call.
 *
 * The wrappers between the GENERATED markers are written by
 * scripts/correctness/lib_v2/gen_v2_runtime.py; edit the generator, not them.
 *===----------------------------------------------------------------------===*/
#include "polygeist_v2_rt.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- error handling ----------------------------------------------------- */

static void polygeist_v2_fatal(const char *what, const char *detail) {
  fprintf(stderr, "polygeist_v2 %s: %s\n", what, detail);
  abort();
}

#define POLYGEIST_V2_CUDA(expr)                                                \
  do {                                                                         \
    cudaError_t polygeist_v2_e = (expr);                                       \
    if (polygeist_v2_e != cudaSuccess)                                         \
      polygeist_v2_fatal(#expr, cudaGetErrorString(polygeist_v2_e));           \
  } while (0)

#define POLYGEIST_V2_CUBLAS(expr, name)                                        \
  do {                                                                         \
    cublasStatus_t polygeist_v2_s = (expr);                                    \
    if (polygeist_v2_s != CUBLAS_STATUS_SUCCESS) {                             \
      char polygeist_v2_buf[64];                                               \
      snprintf(polygeist_v2_buf, sizeof polygeist_v2_buf, "cublas status %d",  \
               (int)polygeist_v2_s);                                           \
      polygeist_v2_fatal(name, polygeist_v2_buf);                              \
    }                                                                          \
  } while (0)

#define POLYGEIST_V2_CUSOLVER(expr, name)                                      \
  do {                                                                         \
    cusolverStatus_t polygeist_v2_s = (expr);                                  \
    if (polygeist_v2_s != CUSOLVER_STATUS_SUCCESS) {                           \
      char polygeist_v2_buf[64];                                               \
      snprintf(polygeist_v2_buf, sizeof polygeist_v2_buf,                      \
               "cusolver status %d", (int)polygeist_v2_s);                     \
      polygeist_v2_fatal(name, polygeist_v2_buf);                              \
    }                                                                          \
  } while (0)

/* ---- handles ------------------------------------------------------------ */

static cublasHandle_t polygeist_v2_cublas_h;
static cusolverDnHandle_t polygeist_v2_cusolver_h;

__attribute__((constructor)) static void polygeist_v2_init(void) {
  POLYGEIST_V2_CUDA(cudaFree(0));
  POLYGEIST_V2_CUBLAS(cublasCreate(&polygeist_v2_cublas_h), "cublasCreate");
  POLYGEIST_V2_CUSOLVER(cusolverDnCreate(&polygeist_v2_cusolver_h),
                        "cusolverDnCreate");
}

static cublasHandle_t polygeist_v2_cublas(void) { return polygeist_v2_cublas_h; }
static cusolverDnHandle_t polygeist_v2_cusolver(void) {
  return polygeist_v2_cusolver_h;
}

/* ---- call accounting (POLYGEIST_V2_TRACE) ------------------------------- */

#define POLYGEIST_V2_MAX_CALLS 64

typedef struct {
  const char *call;
  const char *symbol;
  uint64_t count;
  double h2d_ms, lib_ms, d2h_ms;
  uint64_t h2d_bytes, d2h_bytes;
} polygeist_v2_stat_t;

static polygeist_v2_stat_t polygeist_v2_stats[POLYGEIST_V2_MAX_CALLS];
static int polygeist_v2_nstats;
static int polygeist_v2_trace = -1;

static void polygeist_v2_report(void) {
  for (int i = 0; i < polygeist_v2_nstats; ++i) {
    polygeist_v2_stat_t *s = &polygeist_v2_stats[i];
    fprintf(stderr,
            "POLYGEIST_V2_STATS backend=cuda call=%s entry=%s count=%llu "
            "lib_ms=%.6f h2d_ms=%.6f d2h_ms=%.6f h2d_bytes=%llu "
            "d2h_bytes=%llu\n",
            s->call, s->symbol, (unsigned long long)s->count, s->lib_ms,
            s->h2d_ms, s->d2h_ms, (unsigned long long)s->h2d_bytes,
            (unsigned long long)s->d2h_bytes);
  }
}

static int polygeist_v2_tracing(void) {
  if (polygeist_v2_trace < 0) {
    const char *e = getenv("POLYGEIST_V2_TRACE");
    polygeist_v2_trace = e && *e && strcmp(e, "0") != 0;
    if (polygeist_v2_trace)
      atexit(polygeist_v2_report);
  }
  return polygeist_v2_trace;
}

static polygeist_v2_stat_t *polygeist_v2_stat(const char *call,
                                              const char *symbol) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return &polygeist_v2_stats[i];
  if (polygeist_v2_nstats == POLYGEIST_V2_MAX_CALLS)
    polygeist_v2_fatal(call, "too many distinct calls");
  polygeist_v2_stat_t *s = &polygeist_v2_stats[polygeist_v2_nstats++];
  s->call = call;
  s->symbol = symbol;
  return s;
}

uint64_t polygeist_v2_call_count(const char *call) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return polygeist_v2_stats[i].count;
  return 0;
}

/* ---- per-call host<->device mapping ------------------------------------- */

#define POLYGEIST_V2_MAX_OPERANDS 8
#define POLYGEIST_V2_SLOTS (POLYGEIST_V2_MAX_OPERANDS + 1)

static char *polygeist_v2_slot_ptr[POLYGEIST_V2_SLOTS];
static size_t polygeist_v2_slot_cap[POLYGEIST_V2_SLOTS];

static void *polygeist_v2_slot(int slot, size_t bytes) {
  if (bytes == 0)
    bytes = 1;
  if (polygeist_v2_slot_cap[slot] < bytes) {
    if (polygeist_v2_slot_ptr[slot])
      POLYGEIST_V2_CUDA(cudaFree(polygeist_v2_slot_ptr[slot]));
    POLYGEIST_V2_CUDA(
        cudaMalloc((void **)&polygeist_v2_slot_ptr[slot], bytes));
    polygeist_v2_slot_cap[slot] = bytes;
  }
  return polygeist_v2_slot_ptr[slot];
}

typedef struct {
  const char *lo, *hi;
  int written;
  int interval;
} polygeist_v2_operand_t;

typedef struct {
  const char *lo, *hi;
  int written;
  char *dev;
} polygeist_v2_interval_t;

typedef struct {
  const char *call, *symbol;
  polygeist_v2_operand_t op[POLYGEIST_V2_MAX_OPERANDS];
  int nop;
  polygeist_v2_interval_t iv[POLYGEIST_V2_MAX_OPERANDS];
  int niv;
  cudaEvent_t ev[4];
} polygeist_v2_map_t;

static void polygeist_v2_map_init(polygeist_v2_map_t *m, const char *call,
                                  const char *symbol) {
  memset(m, 0, sizeof *m);
  m->call = call;
  m->symbol = symbol;
  for (int i = 0; i < 4; ++i)
    POLYGEIST_V2_CUDA(cudaEventCreate(&m->ev[i]));
  POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[0], 0));
}

static int polygeist_v2_map_add(polygeist_v2_map_t *m, const void *host,
                                int64_t bytes, int written) {
  if (m->nop == POLYGEIST_V2_MAX_OPERANDS)
    polygeist_v2_fatal(m->call, "too many array operands");
  if (bytes < 0)
    polygeist_v2_fatal(m->call, "negative byte extent");
  polygeist_v2_operand_t *o = &m->op[m->nop];
  o->lo = (const char *)host;
  o->hi = o->lo + bytes;
  o->written = written;
  o->interval = -1;
  return m->nop++;
}

/* Merge overlapping footprints, allocate, copy host->device. */
static void polygeist_v2_map_to_device(polygeist_v2_map_t *m) {
  int order[POLYGEIST_V2_MAX_OPERANDS];
  for (int i = 0; i < m->nop; ++i)
    order[i] = i;
  for (int i = 1; i < m->nop; ++i)
    for (int j = i; j > 0 && m->op[order[j]].lo < m->op[order[j - 1]].lo; --j) {
      int t = order[j];
      order[j] = order[j - 1];
      order[j - 1] = t;
    }
  for (int k = 0; k < m->nop; ++k) {
    polygeist_v2_operand_t *o = &m->op[order[k]];
    if (o->hi == o->lo)
      continue; /* empty view: no device storage */
    polygeist_v2_interval_t *last = m->niv ? &m->iv[m->niv - 1] : NULL;
    if (last && o->lo < last->hi) {
      if (o->hi > last->hi)
        last->hi = o->hi;
      last->written |= o->written;
    } else {
      last = &m->iv[m->niv++];
      last->lo = o->lo;
      last->hi = o->hi;
      last->written = o->written;
    }
    o->interval = m->niv - 1;
  }
  polygeist_v2_stat_t *s = polygeist_v2_stat(m->call, m->symbol);
  for (int i = 0; i < m->niv; ++i) {
    size_t n = (size_t)(m->iv[i].hi - m->iv[i].lo);
    m->iv[i].dev = (char *)polygeist_v2_slot(i, n);
    POLYGEIST_V2_CUDA(
        cudaMemcpy(m->iv[i].dev, m->iv[i].lo, n, cudaMemcpyHostToDevice));
    s->h2d_bytes += n;
  }
  POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[1], 0));
}

static void *polygeist_v2_map_dev(polygeist_v2_map_t *m, int k) {
  polygeist_v2_operand_t *o = &m->op[k];
  if (o->interval < 0)
    return NULL;
  polygeist_v2_interval_t *iv = &m->iv[o->interval];
  return iv->dev + (o->lo - iv->lo);
}

/* Copy written intervals back, release events, account. */
static void polygeist_v2_map_to_host(polygeist_v2_map_t *m) {
  POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[2], 0));
  polygeist_v2_stat_t *s = polygeist_v2_stat(m->call, m->symbol);
  for (int i = 0; i < m->niv; ++i) {
    if (!m->iv[i].written)
      continue;
    size_t n = (size_t)(m->iv[i].hi - m->iv[i].lo);
    POLYGEIST_V2_CUDA(cudaMemcpy((void *)m->iv[i].lo, m->iv[i].dev, n,
                                 cudaMemcpyDeviceToHost));
    s->d2h_bytes += n;
  }
  POLYGEIST_V2_CUDA(cudaEventRecord(m->ev[3], 0));
  POLYGEIST_V2_CUDA(cudaEventSynchronize(m->ev[3]));
  float h2d = 0, lib = 0, d2h = 0;
  POLYGEIST_V2_CUDA(cudaEventElapsedTime(&h2d, m->ev[0], m->ev[1]));
  POLYGEIST_V2_CUDA(cudaEventElapsedTime(&lib, m->ev[1], m->ev[2]));
  POLYGEIST_V2_CUDA(cudaEventElapsedTime(&d2h, m->ev[2], m->ev[3]));
  for (int i = 0; i < 4; ++i)
    POLYGEIST_V2_CUDA(cudaEventDestroy(m->ev[i]));
  s->count++;
  s->h2d_ms += h2d;
  s->lib_ms += lib;
  s->d2h_ms += d2h;
  if (polygeist_v2_tracing())
    fprintf(stderr,
            "POLYGEIST_V2_CALL backend=cuda call=%s entry=%s lib_ms=%.6f "
            "h2d_ms=%.6f d2h_ms=%.6f\n",
            m->call, m->symbol, lib, h2d, d2h);
}

/* cuSOLVER workspace and status word (library-ABI convention). */
static void *polygeist_v2_workspace(size_t bytes) {
  return polygeist_v2_slot(POLYGEIST_V2_SLOTS - 1, bytes);
}

static int *polygeist_v2_devinfo_ptr;

static int *polygeist_v2_devinfo(void) {
  if (!polygeist_v2_devinfo_ptr)
    POLYGEIST_V2_CUDA(
        cudaMalloc((void **)&polygeist_v2_devinfo_ptr, sizeof(int)));
  POLYGEIST_V2_CUDA(cudaMemset(polygeist_v2_devinfo_ptr, 0, sizeof(int)));
  return polygeist_v2_devinfo_ptr;
}

static void polygeist_v2_check_devinfo(const char *call, int *dev) {
  int info = 0;
  POLYGEIST_V2_CUDA(cudaMemcpy(&info, dev, sizeof info, cudaMemcpyDeviceToHost));
  if (info != 0)
    fprintf(stderr, "polygeist_v2 %s: devInfo=%d\n", call, info);
}

/* BEGIN GENERATED WRAPPERS */
void polygeist_v2_cublasDaxpy(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, double *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDaxpy", "cublasDaxpy_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDaxpy_v2(polygeist_v2_cublas(), n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasDaxpy_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

double polygeist_v2_cublasDdot(int32_t n, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDdot", "cublasDdot_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  polygeist_v2_map_to_device(&pv2_map);
  double result = 0;
  POLYGEIST_V2_CUBLAS(cublasDdot_v2(polygeist_v2_cublas(), n, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy, &result), "cublasDdot_v2");
  polygeist_v2_map_to_host(&pv2_map);
  return result;
}

void polygeist_v2_cublasDgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDgemm", "cublasDgemm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDgemm_v2(polygeist_v2_cublas(), (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDgemm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDgemv(int32_t trans, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *x, int64_t x_bytes, int32_t incx, double beta, double *y, int64_t y_bytes, int32_t incy) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDgemv", "cublasDgemv_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDgemv_v2(polygeist_v2_cublas(), (cublasOperation_t)trans, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy), "cublasDgemv_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDger(int32_t m, int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy, double *A, int64_t A_bytes, int32_t lda) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDger", "cublasDger_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 0);
  int pv2_k_y = polygeist_v2_map_add(&pv2_map, y, y_bytes, 0);
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDger_v2(polygeist_v2_cublas(), m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_y), incy, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda), "cublasDger_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDscal(int32_t n, double alpha, double *x, int64_t x_bytes, int32_t incx) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDscal", "cublasDscal_v2");
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDscal_v2(polygeist_v2_cublas(), n, &alpha, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx), "cublasDscal_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsymm(int32_t side, int32_t uplo, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsymm", "cublasDsymm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsymm_v2(polygeist_v2_cublas(), (cublasSideMode_t)side, (cublasFillMode_t)uplo, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsymm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsyr2k", "cublasDsyr2k_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsyr2k_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsyr2k_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDsyrk(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDsyrk", "cublasDsyrk_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDsyrk_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, n, k, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, &beta, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDsyrk_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDtrmm(int32_t side, int32_t uplo, int32_t trans, int32_t diag, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, double *B, int64_t B_bytes, int32_t ldb, double *C, int64_t C_bytes, int32_t ldc) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDtrmm", "cublasDtrmm_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_B = polygeist_v2_map_add(&pv2_map, B, B_bytes, 1);
  int pv2_k_C = polygeist_v2_map_add(&pv2_map, C, C_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDtrmm_v2(polygeist_v2_cublas(), (cublasSideMode_t)side, (cublasFillMode_t)uplo, (cublasOperation_t)trans, (cublasDiagType_t)diag, m, n, &alpha, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_B), ldb, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_C), ldc), "cublasDtrmm_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cublasDtrsv(int32_t uplo, int32_t trans, int32_t diag, int32_t n, const double *A, int64_t A_bytes, int32_t lda, double *x, int64_t x_bytes, int32_t incx) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cublasDtrsv", "cublasDtrsv_v2");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 0);
  int pv2_k_x = polygeist_v2_map_add(&pv2_map, x, x_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  POLYGEIST_V2_CUBLAS(cublasDtrsv_v2(polygeist_v2_cublas(), (cublasFillMode_t)uplo, (cublasOperation_t)trans, (cublasDiagType_t)diag, n, (const double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_x), incx), "cublasDtrsv_v2");
  polygeist_v2_map_to_host(&pv2_map);
}

void polygeist_v2_cusolverDnDpotrf(int32_t uplo, int32_t n, double *A, int64_t A_bytes, int32_t lda) {
  polygeist_v2_map_t pv2_map;
  polygeist_v2_map_init(&pv2_map, "cusolverDnDpotrf", "cusolverDnDpotrf");
  int pv2_k_A = polygeist_v2_map_add(&pv2_map, A, A_bytes, 1);
  polygeist_v2_map_to_device(&pv2_map);
  int Lwork = 0;
  int *devInfo = polygeist_v2_devinfo();
  POLYGEIST_V2_CUSOLVER(cusolverDnDpotrf_bufferSize(polygeist_v2_cusolver(), (cublasFillMode_t)uplo, n, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, &Lwork), "cusolverDnDpotrf_bufferSize");
  double* Workspace = (double*)polygeist_v2_workspace((size_t)Lwork * sizeof(double));
  POLYGEIST_V2_CUSOLVER(cusolverDnDpotrf(polygeist_v2_cusolver(), (cublasFillMode_t)uplo, n, (double*)polygeist_v2_map_dev(&pv2_map, pv2_k_A), lda, Workspace, Lwork, devInfo), "cusolverDnDpotrf");
  polygeist_v2_map_to_host(&pv2_map);
  polygeist_v2_check_devinfo("cusolverDnDpotrf", devInfo);
}

/* END GENERATED WRAPPERS */
