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
void polygeist_v2_cublasDgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDgemv(int32_t trans, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *x, int64_t x_bytes, int32_t incx, double beta, double *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cublasDger(int32_t m, int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, const double *y, int64_t y_bytes, int32_t incy, double *A, int64_t A_bytes, int32_t lda);
void polygeist_v2_cublasDscal(int32_t n, double alpha, double *x, int64_t x_bytes, int32_t incx);
void polygeist_v2_cublasDsymm(int32_t side, int32_t uplo, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, const double *B, int64_t B_bytes, int32_t ldb, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDsyrk(int32_t uplo, int32_t trans, int32_t n, int32_t k, double alpha, const double *A, int64_t A_bytes, int32_t lda, double beta, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDtrmm(int32_t side, int32_t uplo, int32_t trans, int32_t diag, int32_t m, int32_t n, double alpha, const double *A, int64_t A_bytes, int32_t lda, double *B, int64_t B_bytes, int32_t ldb, double *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasDtrsv(int32_t uplo, int32_t trans, int32_t diag, int32_t n, const double *A, int64_t A_bytes, int32_t lda, double *x, int64_t x_bytes, int32_t incx);
void polygeist_v2_cublasScopy(int32_t n, const float *x, int64_t x_bytes, int32_t incx, float *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cublasSgeam(int32_t transa, int32_t transb, int32_t m, int32_t n, float alpha, const float *A, int64_t A_bytes, int32_t lda, float beta, const float *B, int64_t B_bytes, int32_t ldb, float *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasSgemm(int32_t transa, int32_t transb, int32_t m, int32_t n, int32_t k, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *B, int64_t B_bytes, int32_t ldb, float beta, float *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cublasSgemv(int32_t trans, int32_t m, int32_t n, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *x, int64_t x_bytes, int32_t incx, float beta, float *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cudnnBackendExecute_aten_abs_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_add_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_atan2_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_ceil_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_cos_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_div_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_erf_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_exp_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_floor_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_leaky_relu_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float negval);
void polygeist_v2_cudnnBackendExecute_aten_log_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_mul_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_neg_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_pow_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_reciprocal_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_rsqrt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_scaled_add_f32(int32_t n, const float *self, int64_t self_bytes, int32_t self_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float alpha);
void polygeist_v2_cudnnBackendExecute_aten_sigmoid_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sigmoid_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_silu_backward_f32(int32_t n, const float *dy, int64_t dy_bytes, int32_t dy_stride, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_silu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sin_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sqrt_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_tanh_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_tanh_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_threshold_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float threshold, float value);
void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride, float epsilon);
void polygeist_v2_cudnnBackendExecute_ggml_swiglu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *gate, int64_t gate_bytes, int32_t gate_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnSoftmaxForward(int32_t n, const float *x, int64_t x_bytes, int32_t incx, float *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cusolverDnDpotrf(int32_t uplo, int32_t n, double *A, int64_t A_bytes, int32_t lda);
void polygeist_v2_cutensorPermute_aten_acos_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_acosh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_asin_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_asinh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_atan_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_atanh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_cosh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_sinh_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cutensorPermute_aten_tan_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
/* END GENERATED PROTOTYPES */

#ifdef __cplusplus
}
#endif

#endif /* POLYGEIST_V2_RT_H */
