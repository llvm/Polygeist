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

/* Matrix descriptor adaptation; no arithmetic replacement. */
void polygeist_v2_matrix_require_disjoint(const void *, int64_t, const void *, int64_t, int32_t);
void *polygeist_v2_matrix_pack(void *, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t);
void polygeist_v2_matrix_release(void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t);

/* Physical memref.copy: exact bits, strides and offsets; no arithmetic.
 * CUDA runtime accepts host/device/managed pointers and synchronizes movement. */
void polygeist_v2_memref_copy(int64_t rank, int64_t element,
                            const int64_t *sizes, const int64_t *src_strides,
                            const int64_t *dst_strides, const void *source,
                            void *destination);

/* BEGIN GENERATED PROTOTYPES */
void polygeist_v2_DeviceReduce_cub_reduce_product_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_DeviceReduce_cub_reduce_sum_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_DeviceReduce_cub_reduce_sum_f64(int32_t n, double seed, const double *input, int64_t input_bytes, int32_t input_stride, double *output, int64_t output_bytes, int32_t output_stride);
float polygeist_v2_DeviceReduce_cub_reduce_value_sum_f32(int32_t n, float seed, const float *input, int64_t input_bytes, int32_t input_stride);
double polygeist_v2_DeviceReduce_cub_reduce_value_sum_f64(int32_t n, double seed, const double *input, int64_t input_bytes, int32_t input_stride);
void polygeist_v2_DeviceScanInclusiveScanInit_cub_inclusive_product_f32(int32_t n, float inclusive, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_DeviceScanInclusiveScanInit_cub_inclusive_scan_f32(int32_t n, float inclusive, const float *input, int64_t input_bytes, int32_t input_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cublasDaxpy(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t incx, double *y, int64_t y_bytes, int32_t incy);
void polygeist_v2_cublasDgeam_openblas_daxpby_geam_f64(int32_t n, double alpha, const double *x, int64_t x_bytes, int32_t x_stride, double beta, double *y, int64_t y_bytes, int32_t y_stride);
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
void polygeist_v2_cublasSsyr2k(int32_t uplo, int32_t trans, int32_t n, int32_t k, float alpha, const float *A, int64_t A_bytes, int32_t lda, const float *B, int64_t B_bytes, int32_t ldb, float beta, float *C, int64_t C_bytes, int32_t ldc);
void polygeist_v2_cudnnBackendExecute_aten_abs_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_add_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_addcdiv_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *t1_val, int64_t t1_val_bytes, int32_t t1_val_stride, const float *t2_val, int64_t t2_val_bytes, int32_t t2_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float scalar_val);
void polygeist_v2_cudnnBackendExecute_aten_addcmul_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *t1_val, int64_t t1_val_bytes, int32_t t1_val_stride, const float *t2_val, int64_t t2_val_bytes, int32_t t2_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float scalar_val);
void polygeist_v2_cudnnBackendExecute_aten_addr_elementwise_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, const float *vec1_val, int64_t vec1_val_bytes, int32_t vec1_val_stride, const float *vec2_val, int64_t vec2_val_bytes, int32_t vec2_val_stride, float beta_val, float alpha_val, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_atan2_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_ceil_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_clamp_max_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float max);
void polygeist_v2_cudnnBackendExecute_aten_clamp_min_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float min);
void polygeist_v2_cudnnBackendExecute_aten_clamp_scalar_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float min, float max);
void polygeist_v2_cudnnBackendExecute_aten_cos_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_div_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_eq_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_erf_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_exp_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_floor_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_frac_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_ge_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_gelu_exact_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_gelu_tanh_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_gt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_hardsigmoid_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_hardsigmoid_f32(int32_t n, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_hardswish_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_hardswish_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_hardtanh_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float min_val, float max_val);
void polygeist_v2_cudnnBackendExecute_aten_le_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_leaky_relu_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float negval);
void polygeist_v2_cudnnBackendExecute_aten_log_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_log_sigmoid_backward_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *buffer, int64_t buffer_bytes, int32_t buffer_stride, const float *grad, int64_t grad_bytes, int32_t grad_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_logical_and_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_logical_not_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_logical_or_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_logical_xor_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_lt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_mul_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_ne_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_neg_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_pow_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_pow_uniform_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float exponent, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_reciprocal_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_rsqrt_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_scaled_add_f32(int32_t n, const float *self, int64_t self_bytes, int32_t self_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float alpha);
void polygeist_v2_cudnnBackendExecute_aten_shrink_backward_f32(int32_t n, const float *grad_val, int64_t grad_val_bytes, int32_t grad_val_stride, const float *self_val, int64_t self_val_bytes, int32_t self_val_stride, float *output, int64_t output_bytes, int32_t output_stride, float lambd_val);
void polygeist_v2_cudnnBackendExecute_aten_sigmoid_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sigmoid_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sign_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_silu_backward_f32(int32_t n, const float *dy, int64_t dy_bytes, int32_t dy_stride, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_silu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_sin_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_softshrink_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride, float lambd_val);
void polygeist_v2_cudnnBackendExecute_aten_sqrt_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_tanh_backward_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, const float *b, int64_t b_bytes, int32_t b_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_tanh_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_aten_threshold_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *other, int64_t other_bytes, int32_t other_stride, float *output, int64_t output_bytes, int32_t output_stride, float threshold, float value);
void polygeist_v2_cudnnBackendExecute_aten_trunc_f32(int32_t n, const float *a, int64_t a_bytes, int32_t a_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnBackendExecute_cutlass_rmsnorm_parameterized_f32(int32_t n, const float *input, int64_t input_bytes, int32_t input_stride, const float *weight, int64_t weight_bytes, int32_t weight_stride, float *output, int64_t output_bytes, int32_t output_stride, float epsilon);
void polygeist_v2_cudnnBackendExecute_ggml_swiglu_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *gate, int64_t gate_bytes, int32_t gate_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_cudnnSoftmaxBackward(int32_t n, const float *dy, int64_t dy_bytes, int32_t incdy, const float *y, int64_t y_bytes, int32_t incy, float *dx, int64_t dx_bytes, int32_t incdx);
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
void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_s16(int32_t h, int32_t w, const short *input, int64_t input_bytes, int32_t input_stride, short *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_s8(int32_t h, int32_t w, const signed char *input, int64_t input_bytes, int32_t input_stride, signed char *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_u16(int32_t h, int32_t w, const unsigned short *input, int64_t input_bytes, int32_t input_stride, unsigned short *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_u8(int32_t h, int32_t w, const unsigned char *input, int64_t input_bytes, int32_t input_stride, unsigned char *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_transform_thrust_divide_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_transform_thrust_divide_scalar_left_f32(int32_t n, float x, const float *y, int64_t y_bytes, int32_t y_stride, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_transform_thrust_divide_scalar_right_f32(int32_t n, const float *x, int64_t x_bytes, int32_t x_stride, float y, float *output, int64_t output_bytes, int32_t output_stride);
void polygeist_v2_transform_thrust_divide_scalars_f32(int32_t n, float x, float y, float *output, int64_t output_bytes, int32_t output_stride);
/* END GENERATED PROTOTYPES */

#ifdef __cplusplus
}
#endif

#endif /* POLYGEIST_V2_RT_H */
