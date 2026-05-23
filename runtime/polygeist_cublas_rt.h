// polygeist_cublas_rt.h — runtime shim ABI for the
// `--lower-kernel-launch-to-cublas` pass.
//
// The pass emits `func.call` ops targeting these C functions. The functions
// are implemented in two flavours:
//   * polygeist_cublas_rt_cpu.c   — reference CPU implementation (no CUDA).
//                                    Used for correctness validation on
//                                    machines without a GPU.
//   * polygeist_cublas_rt_cuda.c  — real cuBLAS implementation. Used on
//                                    Jetson / x86 + NVIDIA GPU.
// Link exactly one of them into the executable.
//
// All matrices are ROW-MAJOR f64. Leading dimensions are in elements
// (not bytes). The CUDA backend internally does the row↔col-major dance
// (compute Cᵀ = BᵀAᵀ via operand swap) so callers can stay row-major.
//
// Sizes are passed as int32_t because that matches cuBLAS's signature.

#ifndef POLYGEIST_CUBLAS_RT_H
#define POLYGEIST_CUBLAS_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle. Call init() once before any kernel calls; destroy() at exit.
// On CPU these are no-ops; on CUDA they create a cublasHandle_t + stream.
void polygeist_cublas_init(void);
void polygeist_cublas_destroy(void);

// GEMM (cublasDgemm equivalent, row-major):
//   C = alpha * A * B + beta * C
// where A is MxK, B is KxN, C is MxN.
//
// For non-transposed inputs at row-major:
//   lda = K, ldb = N, ldc = N.
//
// On CUDA: copies A/B/C H→D, calls cublasDgemm with operand swap to handle
// the row→col-major transpose, copies C D→H, frees device buffers. Each call
// is fully synchronous; device-residency hoisting is a follow-up.
void polygeist_cublas_dgemm(
    int32_t M, int32_t N, int32_t K,
    double alpha,
    const double *A, int32_t lda,
    const double *B, int32_t ldb,
    double beta,
    double *C, int32_t ldc);

// memset a 2D row-major MxN block to zero. Used by matcher's
// @memset_zero_2D op. Trivial host-side memset; data is host-resident
// between launches in the current no-hoisting model.
void polygeist_cublas_memset_zero_2d(
    int32_t M, int32_t N, double *A, int32_t lda);

// In-place 2D scale: A = scale * A, row-major MxN with leading dim lda.
// Used by matcher's @cublasDgeam_scale2D op (the diagonal/scale-only
// variant of geam where the second operand is zero so the add collapses
// to a scale). CUDA backend uses cublasDscal on the flattened buffer
// when contiguous (lda==N), else loops row-wise.
void polygeist_cublas_dscal_2d(
    int32_t M, int32_t N, double scale, double *A, int32_t lda);

// cuDNN 9-tap conv2d (3x3 stencil) with PolyBench's hardcoded weights.
// Input A is MxN row-major f64; output B is MxN row-major f64; the
// interior B[1..M-2][1..N-2] is filled with the convolved result,
// border rows/cols are untouched. CUDA backend calls cudnnConvolutionForward
// with a 1×1×M×N input descriptor and a 1×1×3×3 filter descriptor.
// CPU stub does the same math in a 3-loop reference for validation.
//
// Weights baked in (matches polybenchGpu/OpenMP/stencils/convolution-2d/):
//   [[ 0.2, 0.5, -0.8],
//    [-0.3, 0.6, -0.9],
//    [ 0.4, 0.7,  0.1]]
//
// Generalising the weights to arbitrary filter coefficients is a TODO
// once the matcher surfaces the 9 scalar weights as launch operands.
void polygeist_cudnn_conv2d_polybench9tap(
    int32_t M, int32_t N, const double *A, double *B);

// Generic 3x3 conv2d shim — takes the 9 filter weights at runtime so a
// single shim handles any 3x3 weighted conv (polybench, Sobel, Gaussian,
// custom filters). Same I/O contract as the polybench9tap variant:
//   * A is MxN row-major f64, input
//   * B is MxN row-major f64, output; interior B[1..M-2][1..N-2] written
//   * Weights laid out row-major in the 3x3 filter:
//       w[0] w[1] w[2]      <- top row, applied to A[i-1][j-1..j+1]
//       w[3] w[4] w[5]      <- middle row, applied to A[i][j-1..j+1]
//       w[6] w[7] w[8]      <- bottom row, applied to A[i+1][j-1..j+1]
//
// Used by Lit-surfaced @cudnnConvolution2D_9tap match: the matcher pulls
// the 9 weight values out of the linalg.generic body and passes them as
// launch operands, the lowering pass forwards them here.
void polygeist_cudnn_conv2d_3x3_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2,
    double w3, double w4, double w5,
    double w6, double w7, double w8,
    const double *A, double *B);

// FP32 variant of polygeist_cudnn_conv2d_3x3 — same I/O contract but with
// float matrices + float weights. cuDNN's convolution path picks tensor-core
// kernels for FP32 on Ampere+ GPUs (including Jetson Orin), so this is the
// dtype to use for actual perf measurement (FP64 on Orin uses a generic
// non-tensor-core path).
void polygeist_cudnn_conv2d_3x3_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2,
    float w3, float w4, float w5,
    float w6, float w7, float w8,
    const float *A, float *B);

// FP16 / BF16 variants. The shim args use compiler-provided half-precision
// types (`_Float16` for IEEE half, `__bf16` for brain-float) because MLIR's
// `f16` / `bf16` lower to LLVM `half` / `bfloat` and use the FP-register ABI
// on both x86-64 (XMM) and aarch64 (V regs). Passing them via uint16_t would
// route through GP regs and corrupt the call.
//   * f16  → CUDNN_DATA_HALF      (cuDNN tensor-core path on Ampere+)
//   * bf16 → CUDNN_DATA_BFLOAT16  (tensor-core path on Ampere+)
// Guarded on compiler-defined feature macros: __FLT16_MAX__ for `_Float16`
// and __BFLT16_MAX__ for `__bf16`. Both are defined unconditionally on
// aarch64 (Jetson) and on x86-64 when the appropriate -m flags are set
// (-mavx512fp16 / -mavx512bf16). If a build target lacks the macro the
// declaration is skipped — callers can't accidentally link to a missing
// symbol because the shim implementation file is guarded the same way.
#if defined(__FLT16_MAX__)
void polygeist_cudnn_conv2d_3x3_f16(
    int32_t M, int32_t N,
    _Float16 w0, _Float16 w1, _Float16 w2,
    _Float16 w3, _Float16 w4, _Float16 w5,
    _Float16 w6, _Float16 w7, _Float16 w8,
    const _Float16 *A, _Float16 *B);
#endif

#if defined(__BFLT16_MAX__) || defined(__ARM_FEATURE_BF16) || \
    defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || defined(__BF16__)
void polygeist_cudnn_conv2d_3x3_bf16(
    int32_t M, int32_t N,
    __bf16 w0, __bf16 w1, __bf16 w2,
    __bf16 w3, __bf16 w4, __bf16 w5,
    __bf16 w6, __bf16 w7, __bf16 w8,
    const __bf16 *A, __bf16 *B);
#endif

// INT32 / INT16 variants.
//
// IMPORTANT: cuDNN does NOT support a standalone INT32 forward convolution
// (`cudnnSetTensor4dDescriptor` with CUDNN_DATA_INT32 returns BAD_PARAM on
// Orin/Ampere). CUDNN_DATA_INT32 is only exposed as the accumulator type
// for INT8 inputs via the bias+activation API — a different operand
// layout. Consequently the CUDA backend's i32 / i16 shims intentionally
// fail at the cuDNN descriptor call: they exist so the matcher /
// rewriter / ABI-lowering pipeline can be exercised end-to-end (the
// `func.call @polygeist_cudnn_conv2d_3x3_i32` will land), but the GPU
// side is "not implemented" until a custom CUDA kernel is added.
//
// The CPU backend's i32 / i16 implementations are real reference loops;
// use the CPU stub for correctness validation of int conv stencils.
void polygeist_cudnn_conv2d_3x3_i32(
    int32_t M, int32_t N,
    int32_t w0, int32_t w1, int32_t w2,
    int32_t w3, int32_t w4, int32_t w5,
    int32_t w6, int32_t w7, int32_t w8,
    const int32_t *A, int32_t *B);

void polygeist_cudnn_conv2d_3x3_i16(
    int32_t M, int32_t N,
    int16_t w0, int16_t w1, int16_t w2,
    int16_t w3, int16_t w4, int16_t w5,
    int16_t w6, int16_t w7, int16_t w8,
    const int16_t *A, int16_t *B);

// Per-call CUDA-event timing (CUDA backend only — CPU stub returns 0.0).
// Pair with polygeist_cublas_time_begin / polygeist_cublas_time_end around
// a sequence of kernel calls.
void  polygeist_cublas_time_begin(void);
double polygeist_cublas_time_end_ms(void);  // returns ms since last begin

#ifdef __cplusplus
}
#endif

#endif  // POLYGEIST_CUBLAS_RT_H
