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

// Per-call CUDA-event timing (CUDA backend only — CPU stub returns 0.0).
// Pair with polygeist_cublas_time_begin / polygeist_cublas_time_end around
// a sequence of kernel calls.
void  polygeist_cublas_time_begin(void);
double polygeist_cublas_time_end_ms(void);  // returns ms since last begin

#ifdef __cplusplus
}
#endif

#endif  // POLYGEIST_CUBLAS_RT_H
