/*===- polygeist_v2_rt_cpu.c - lib_v2 ABI on Fortran BLAS/LAPACK ----------===*
 *
 * CPU implementation of polygeist_v2_rt.h.  Every wrapper forwards to the
 * reference Fortran entry point (`dgemm_`, `dtrsv_`, `dpotrf_`, ...) of
 * whatever BLAS/LAPACK is linked (the build links OpenBLAS, `-lopenblas`, as
 * polygeist_build.sh does for POLYGEIST_CPU_BLAS=1; OpenBLAS also exports the
 * LAPACK symbols).
 *
 * Fortran calling convention (gfortran/OpenBLAS LP64): every argument by
 * reference, INTEGER = int32_t, CHARACTER*1 = char* plus one hidden size_t
 * length per CHARACTER argument appended after the explicit arguments.
 *
 * The wrappers between the GENERATED markers are written by
 * scripts/correctness/lib_v2/gen_v2_runtime.py; edit the generator, not them.
 *===----------------------------------------------------------------------===*/
#define _POSIX_C_SOURCE 199309L
#include "polygeist_v2_rt.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- call accounting (POLYGEIST_V2_TRACE) ------------------------------- */

#define POLYGEIST_V2_MAX_CALLS 64

typedef struct {
  const char *call;
  const char *entry;
  uint64_t count;
  double ms;
} polygeist_v2_stat_t;

static polygeist_v2_stat_t polygeist_v2_stats[POLYGEIST_V2_MAX_CALLS];
static int polygeist_v2_nstats;
static int polygeist_v2_trace = -1;

static void polygeist_v2_report(void) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    fprintf(stderr,
            "POLYGEIST_V2_STATS backend=cpu call=%s entry=%s count=%llu "
            "lib_ms=%.6f\n",
            polygeist_v2_stats[i].call, polygeist_v2_stats[i].entry,
            (unsigned long long)polygeist_v2_stats[i].count,
            polygeist_v2_stats[i].ms);
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
                                              const char *entry) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (polygeist_v2_stats[i].call == call ||
        strcmp(polygeist_v2_stats[i].call, call) == 0)
      return &polygeist_v2_stats[i];
  if (polygeist_v2_nstats == POLYGEIST_V2_MAX_CALLS) {
    fprintf(stderr, "polygeist_v2: too many distinct calls\n");
    abort();
  }
  polygeist_v2_stat_t *s = &polygeist_v2_stats[polygeist_v2_nstats++];
  s->call = call;
  s->entry = entry;
  return s;
}

uint64_t polygeist_v2_call_count(const char *call) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return polygeist_v2_stats[i].count;
  return 0;
}

static double polygeist_v2_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void polygeist_v2_done(const char *call, const char *entry,
                              double t0) {
  double dt = polygeist_v2_now_ms() - t0;
  polygeist_v2_stat_t *s = polygeist_v2_stat(call, entry);
  s->count++;
  s->ms += dt;
  if (polygeist_v2_tracing())
    fprintf(stderr, "POLYGEIST_V2_CALL backend=cpu call=%s entry=%s ms=%.6f\n",
            call, entry, dt);
}

static void polygeist_v2_fatal(const char *call, const char *msg) {
  fprintf(stderr, "polygeist_v2 %s: %s\n", call, msg);
  abort();
}

static void polygeist_v2_info(const char *call, const char *entry, int info) {
  if (info != 0)
    fprintf(stderr, "polygeist_v2 %s: %s returned INFO=%d\n", call, entry,
            info);
}

/* A target parameter the Fortran entry point does not have (e.g. the
 * out-of-place result of cuBLAS's trmm) is only accepted when it aliases an
 * argument the Fortran routine updates in place, with the same layout. */
static void polygeist_v2_require_alias(const char *call, const void *p,
                                       int32_t ld, const void *const *cands,
                                       const int32_t *cand_ld, int n) {
  for (int i = 0; i < n; ++i)
    if (cands[i] == p && cand_ld[i] == ld)
      return;
  polygeist_v2_fatal(call, "out-of-place operand has no in-place Fortran "
                           "equivalent (it must alias an updated argument)");
}

typedef size_t polygeist_v2_flen;

/* BEGIN GENERATED WRAPPERS */
extern void daxpy_(const int *, const double *, const double *, const int *, double *, const int *);
void polygeist_v2_cublasDaxpy(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, double *y, int64_t y_bytes, int32_t incy) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  (void)x_bytes;
  (void)y_bytes;
  daxpy_(&n, &alpha, x, &incx, y, &incy);
  polygeist_v2_done("cublasDaxpy", "daxpy_", polygeist_v2_t0);
}

extern double ddot_(const int *, const double *, const int *, const double *, const int *);
double polygeist_v2_cublasDdot(int32_t n, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  (void)x_bytes;
  (void)y_bytes;
  double polygeist_v2_r = ddot_(&n, x, &incx, y, &incy);
  polygeist_v2_done("cublasDdot", "ddot_", polygeist_v2_t0);
  return polygeist_v2_r;
}

extern void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDgemm_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDgemm", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_transa = polygeist_v2_cublasDgemm_cublasOperation_t_char(transa);
  char pv2_c_transb = polygeist_v2_cublasDgemm_cublasOperation_t_char(transb);
  (void)A_bytes;
  (void)B_bytes;
  (void)C_bytes;
  dgemm_(&pv2_c_transa, &pv2_c_transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDgemm", "dgemm_", polygeist_v2_t0);
}

extern void dgemv_(const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *, polygeist_v2_flen);
static char polygeist_v2_cublasDgemv_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDgemv", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDgemv(int32_t trans, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *x, int64_t x_bytes, int32_t incx, double beta, double *y, int64_t y_bytes, int32_t incy) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_trans = polygeist_v2_cublasDgemv_cublasOperation_t_char(trans);
  (void)A_bytes;
  (void)x_bytes;
  (void)y_bytes;
  dgemv_(&pv2_c_trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDgemv", "dgemv_", polygeist_v2_t0);
}

extern void dger_(const int *, const int *, const double *, const double *, const int *, const double *, const int *, double *, const int *);
void polygeist_v2_cublasDger(int32_t m, int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy, double *A, int64_t A_bytes, int32_t lda) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  (void)x_bytes;
  (void)y_bytes;
  (void)A_bytes;
  dger_(&m, &n, &alpha, x, &incx, y, &incy, A, &lda);
  polygeist_v2_done("cublasDger", "dger_", polygeist_v2_t0);
}

extern void dscal_(const int *, const double *, double *, const int *);
void polygeist_v2_cublasDscal(int32_t n, double alpha, double *x, int64_t x_bytes, int32_t incx) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  (void)x_bytes;
  dscal_(&n, &alpha, x, &incx);
  polygeist_v2_done("cublasDscal", "dscal_", polygeist_v2_t0);
}

extern void dsymm_(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDsymm_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDsymm", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDsymm_cublasSideMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'R';
  }
  polygeist_v2_fatal("cublasDsymm", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDsymm(int32_t side, int32_t uplo, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_side = polygeist_v2_cublasDsymm_cublasSideMode_t_char(side);
  char pv2_c_uplo = polygeist_v2_cublasDsymm_cublasFillMode_t_char(uplo);
  (void)A_bytes;
  (void)B_bytes;
  (void)C_bytes;
  dsymm_(&pv2_c_side, &pv2_c_uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDsymm", "dsymm_", polygeist_v2_t0);
}

extern void dsyr2k_(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDsyr2k_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDsyr2k", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDsyr2k_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDsyr2k", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_uplo = polygeist_v2_cublasDsyr2k_cublasFillMode_t_char(uplo);
  char pv2_c_trans = polygeist_v2_cublasDsyr2k_cublasOperation_t_char(trans);
  (void)A_bytes;
  (void)B_bytes;
  (void)C_bytes;
  dsyr2k_(&pv2_c_uplo, &pv2_c_trans, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDsyr2k", "dsyr2k_", polygeist_v2_t0);
}

extern void dsyrk_(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDsyrk_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDsyrk", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDsyrk_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDsyrk", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDsyrk(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, double beta, double *C, int64_t C_bytes, int32_t ldc) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_uplo = polygeist_v2_cublasDsyrk_cublasFillMode_t_char(uplo);
  char pv2_c_trans = polygeist_v2_cublasDsyrk_cublasOperation_t_char(trans);
  (void)A_bytes;
  (void)C_bytes;
  dsyrk_(&pv2_c_uplo, &pv2_c_trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDsyrk", "dsyrk_", polygeist_v2_t0);
}

extern void dtrmm_(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *, const int *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDtrmm_cublasDiagType_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDtrmm", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDtrmm_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDtrmm", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDtrmm_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDtrmm", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDtrmm_cublasSideMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'R';
  }
  polygeist_v2_fatal("cublasDtrmm", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDtrmm(int32_t side, int32_t uplo, int32_t trans, int32_t diag, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, double *B, int64_t B_bytes, int32_t ldb, double *C, int64_t C_bytes, int32_t ldc) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_side = polygeist_v2_cublasDtrmm_cublasSideMode_t_char(side);
  char pv2_c_uplo = polygeist_v2_cublasDtrmm_cublasFillMode_t_char(uplo);
  char pv2_c_trans = polygeist_v2_cublasDtrmm_cublasOperation_t_char(trans);
  char pv2_c_diag = polygeist_v2_cublasDtrmm_cublasDiagType_t_char(diag);
  { const void *const c[] = {B};
    const int32_t l[] = {ldb};
    polygeist_v2_require_alias("cublasDtrmm", C, ldc, c, l, 1); }
  (void)A_bytes;
  (void)B_bytes;
  (void)C_bytes;
  dtrmm_(&pv2_c_side, &pv2_c_uplo, &pv2_c_trans, &pv2_c_diag, &m, &n, &alpha, A, &lda, B, &ldb, (polygeist_v2_flen)1, (polygeist_v2_flen)1, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDtrmm", "dtrmm_", polygeist_v2_t0);
}

extern void dtrsv_(const char *, const char *, const char *, const int *, const double *, const int *, double *, const int *, polygeist_v2_flen, polygeist_v2_flen, polygeist_v2_flen);
static char polygeist_v2_cublasDtrsv_cublasDiagType_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDtrsv", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDtrsv_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cublasDtrsv", "enum value without a Fortran flag");
  return 0;
}
static char polygeist_v2_cublasDtrsv_cublasOperation_t_char(int32_t v) {
  switch (v) {
  case 0: return 'N';
  case 1: return 'T';
  }
  polygeist_v2_fatal("cublasDtrsv", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cublasDtrsv(int32_t uplo, int32_t trans, int32_t diag, int32_t n, const double *A, int64_t A_bytes, int32_t lda, double *x, int64_t x_bytes, int32_t incx) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_uplo = polygeist_v2_cublasDtrsv_cublasFillMode_t_char(uplo);
  char pv2_c_trans = polygeist_v2_cublasDtrsv_cublasOperation_t_char(trans);
  char pv2_c_diag = polygeist_v2_cublasDtrsv_cublasDiagType_t_char(diag);
  (void)A_bytes;
  (void)x_bytes;
  dtrsv_(&pv2_c_uplo, &pv2_c_trans, &pv2_c_diag, &n, A, &lda, x, &incx, (polygeist_v2_flen)1, (polygeist_v2_flen)1, (polygeist_v2_flen)1);
  polygeist_v2_done("cublasDtrsv", "dtrsv_", polygeist_v2_t0);
}

extern void dpotrf_(const char *, const int *, double *, const int *, int *, polygeist_v2_flen);
static char polygeist_v2_cusolverDnDpotrf_cublasFillMode_t_char(int32_t v) {
  switch (v) {
  case 0: return 'L';
  case 1: return 'U';
  }
  polygeist_v2_fatal("cusolverDnDpotrf", "enum value without a Fortran flag");
  return 0;
}
void polygeist_v2_cusolverDnDpotrf(int32_t uplo, int32_t n, double *A, int64_t A_bytes, int32_t lda) {
  double polygeist_v2_t0 = polygeist_v2_now_ms();
  char pv2_c_uplo = polygeist_v2_cusolverDnDpotrf_cublasFillMode_t_char(uplo);
  int pv2_info = 0;
  (void)A_bytes;
  dpotrf_(&pv2_c_uplo, &n, A, &lda, &pv2_info, (polygeist_v2_flen)1);
  polygeist_v2_info("cusolverDnDpotrf", "dpotrf_", pv2_info);
  polygeist_v2_done("cusolverDnDpotrf", "dpotrf_", polygeist_v2_t0);
}

/* END GENERATED WRAPPERS */
