/*===- polygeist_v2_rt.h - lib_v2 library-call ABI ------------------------===*
 *
 * ABI between `lower_v2.py` (kernel.launch @netlib_* -> func.call) and the
 * two runtimes:
 *   polygeist_v2_rt_cpu.c   Fortran BLAS/LAPACK entry points (OpenBLAS etc.)
 *   polygeist_v2_rt_cuda.c  cuBLAS / cuSOLVER pass-through
 *
 * One function per library call named in a lib_v2 manifest
 * (`target.call`), `polygeist_v2_<call>`.  Its parameters are the target's
 * parameters in the order of the vendor header (library handle and
 * runtime-managed workspace/status/result parameters removed):
 *   enum   -> int32_t, value of the vendor enum constant (cublasOperation_t...)
 *   int    -> int32_t
 *   scalar -> double, by value (host)
 *   array  -> pointer to element 0 of the view, followed by int64_t byte
 *             extent of the view's memory footprint
 *             ((sum_d (size_d - 1) * stride_d + 1) * sizeof(elem), 0 if empty)
 * Matrices are column-major with leading dimension = the memref row stride.
 *
 * The prototypes between the GENERATED markers are written by
 * scripts/correctness/lib_v2/gen_v2_runtime.py from abi_v2.py's table
 * (manifests + vendor headers + netlib sources); do not edit them by hand.
 *
 * Observability: if the environment variable POLYGEIST_V2_TRACE is set, each
 * runtime prints one `POLYGEIST_V2_CALL` line per call and a
 * `POLYGEIST_V2_STATS` summary at exit on stderr (outside PolyBench dumps).
 *===----------------------------------------------------------------------===*/
#ifndef POLYGEIST_V2_RT_H
#define POLYGEIST_V2_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of completed calls of `polygeist_v2_<call>` in this process. */
uint64_t polygeist_v2_call_count(const char *call);

/* BEGIN GENERATED PROTOTYPES */
void polygeist_v2_cublasDaxpy(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, double *y, int64_t y_bytes, int32_t incy);
double polygeist_v2_cublasDdot(int32_t n, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cublasDgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDgemv(int32_t trans, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *x, int64_t x_bytes, int32_t incx, double beta, double *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cublasDger(int32_t m, int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy, double *A, int64_t A_bytes, int32_t lda);
void polygeist_v2_cublasDscal(int32_t n, double alpha, double *x, int64_t x_bytes, int32_t incx);
void polygeist_v2_cublasDsymm(int32_t side, int32_t uplo, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDsyrk(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDtrmm(int32_t side, int32_t uplo, int32_t trans, int32_t diag, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, double *B, int64_t B_bytes, int32_t ldb, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDtrsv(int32_t uplo, int32_t trans, int32_t diag, int32_t n, const double *A, int64_t A_bytes, int32_t lda, double *x, int64_t x_bytes, int32_t incx);
void polygeist_v2_cusolverDnDpotrf(int32_t uplo, int32_t n, double *A, int64_t A_bytes, int32_t lda);
/* END GENERATED PROTOTYPES */

#ifdef __cplusplus
}
#endif

#endif /* POLYGEIST_V2_RT_H */
