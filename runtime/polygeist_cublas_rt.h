// polygeist_cublas_rt.h — runtime shim ABI for the
// `--lower-kernel-launch-to-cublas` pass.
//
// The pass emits `func.call` ops targeting these C functions. The functions
// are implemented in two flavours:
//   * polygeist_cublas_rt_cpu.c   — CPU implementation (no CUDA). Reference
//                                    loops by default, or optimized CBLAS for
//                                    BLAS-like symbols when compiled with
//                                    POLYGEIST_CPU_USE_CBLAS.
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

// Pipeline scope. The compiler inserts these around a group/function containing
// lowered library calls. They are currently conservative hooks: CPU is a no-op,
// CUDA initializes the backend on begin and synchronizes at outermost end.
// Future runtime cache/device-residency policies should hang off this scope
// instead of changing every per-kernel shim ABI.
void polygeist_cublas_pipeline_begin(void);
void polygeist_cublas_pipeline_end(void);

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

void polygeist_cublas_sgemm(
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float *A, int32_t lda,
    const float *B, int32_t ldb,
    float beta,
    float *C, int32_t ldc);

void polygeist_cublas_dgemv(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y);

void polygeist_cublas_dgemv_T(
    int32_t M, int32_t N,
    double alpha,
    const double *A, int32_t lda,
    const double *x,
    double beta,
    double *y);

void polygeist_cublas_sgemv(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y);

void polygeist_cublas_sgemv_T(
    int32_t M, int32_t N,
    float alpha,
    const float *A, int32_t lda,
    const float *x,
    float beta,
    float *y);

// FP32 variant of memset_zero_2d.
void polygeist_cublas_memset_zero_2d_f32(
    int32_t M, int32_t N, float *A, int32_t lda);

void polygeist_cublas_memset_zero_1d_f32(int32_t N, float *v);

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

// Generic 5x5 conv2d shim. The lowering passes a pointer to the top-left
// input subview and a pointer to the output interior subview B[2][2], so the
// shim writes a dense (M-4)x(N-4) block relative to B with row stride N.
void polygeist_cudnn_conv2d_5x5_f64(
    int32_t M, int32_t N,
    double w0, double w1, double w2, double w3, double w4,
    double w5, double w6, double w7, double w8, double w9,
    double w10, double w11, double w12, double w13, double w14,
    double w15, double w16, double w17, double w18, double w19,
    double w20, double w21, double w22, double w23, double w24,
    const double *A, double *B);

void polygeist_cudnn_conv2d_5x5_f32(
    int32_t M, int32_t N,
    float w0, float w1, float w2, float w3, float w4,
    float w5, float w6, float w7, float w8, float w9,
    float w10, float w11, float w12, float w13, float w14,
    float w15, float w16, float w17, float w18, float w19,
    float w20, float w21, float w22, float w23, float w24,
    const float *A, float *B);

// Generalized packed-weight odd-square Conv2D stencil. K is the filter width;
// W has K*K row-major weights. A points at the top-left input subview and B
// points at the output interior subview.
void polygeist_cudnn_conv2d_ntap_f64(
    int32_t M, int32_t N, int32_t K,
    const double *W, const double *A, double *B);

void polygeist_cudnn_conv2d_ntap_f32(
    int32_t M, int32_t N, int32_t K,
    const float *W, const float *A, float *B);

// Generalized packed-weight Conv3D stencil. A is dense input with dimensions
// inD x inH x inW, B is dense output with dimensions outD x outH x outW, and
// W is a row-major K x K x K cross-correlation filter.
void polygeist_cudnn_conv3d_ntap_f64(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const double *W, const double *A, double *B);

void polygeist_cudnn_conv3d_ntap_f32(
    int32_t inD, int32_t inH, int32_t inW,
    int32_t outD, int32_t outH, int32_t outW,
    int32_t K,
    const float *W, const float *A, float *B);

// Custom structured 3D 7-point stencil over seven flattened tap tensors.
// `extra` and `coeff` may be NULL. The computation is:
//   base = base0 * a0 + base_extra * extra
//   inner = c0*a0 + c1*a1 + ... + c6*a6 + coeff_extra * extra
//   out = base + (coeff ? coeff[i] : 1) * inner
void polygeist_custom_stencil3d_7pt_flat_f64(
    int32_t N,
    const double *a0, const double *a1, const double *a2,
    const double *a3, const double *a4, const double *a5,
    const double *a6, const double *extra, const double *coeff,
    double *out,
    double base0, double base_extra, double coeff_extra,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6);

void polygeist_custom_stencil3d_7pt_flat_f32(
    int32_t N,
    const float *a0, const float *a1, const float *a2,
    const float *a3, const float *a4, const float *a5,
    const float *a6, const float *extra, const float *coeff,
    float *out,
    float base0, float base_extra, float coeff_extra,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6);

// Basic 1D complex-to-complex FFT shims. Complex values are represented as
// interleaved real/imag pairs: A[2*i+0], A[2*i+1]. `inverse != 0` selects the
// inverse transform. Like cuFFT, the inverse is not normalized by N.
void polygeist_cufft_z2z_1d(
    int32_t N, int32_t inverse, const double *A, double *B);

void polygeist_cufft_c2c_1d(
    int32_t N, int32_t inverse, const float *A, float *B);

// Separable 3D tensor product, expressed to cuTensorNet as
// ai,bj,ck,ijk->abc. psi is row-major [KQ,KP], u is [KP,KP,KP], and out is
// [KQ,KQ,KQ].
void polygeist_cutensornet_tensor_product_3d_f32(
    int32_t KQ, int32_t KP, const float *psi, const float *u, float *out);

void polygeist_cutensornet_tensor_product_3d_f64(
    int32_t KQ, int32_t KP, const double *psi, const double *u, double *out);

// General two-input FP64 Einstein contraction used by normalized MFEM stages.
// `metadata` contains three ranks followed by extent/stride/mode arrays for
// A, B, and C; see LowerKernelLaunchToCuBLAS.cpp for the fixed 48-element
// layout. Broadcast modes are omitted before this ABI is called.
void polygeist_cutensornet_contraction2_f64(
    const double *A, const double *B, double *C, const int64_t *metadata);

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

// PVA-routed INT8 / INT16 conv (NEW path; replaces the failing-cuDNN i8/i16
// shims for the lowering). Same I/O contract as the cuDNN 3x3 shims:
//   - A, B are MxN row-major buffers of int8_t / int16_t
//   - Interior B[1..M-2][1..N-2] gets the convolved result; borders left untouched
// Routes via PVA Solutions' pvaConv2d through libpva_operator.so on the Jetson.
// PVA's conv supports kernel 3x3/5x5/7x7, single-channel, integer 8/16-bit,
// with an internal wider accumulator + output narrowing. CPU stub does a
// reference loop with int32 accumulator and narrowing-with-wrap on
// store (matches PVA's behaviour for our polybench-scaled weights since
// the per-pixel sum stays in narrow-int range).
void polygeist_pva_conv2d_3x3_i8(
    int32_t M, int32_t N,
    int8_t w0, int8_t w1, int8_t w2,
    int8_t w3, int8_t w4, int8_t w5,
    int8_t w6, int8_t w7, int8_t w8,
    const int8_t *A, int8_t *B);

void polygeist_pva_conv2d_3x3_i16(
    int32_t M, int32_t N,
    int16_t w0, int16_t w1, int16_t w2,
    int16_t w3, int16_t w4, int16_t w5,
    int16_t w6, int16_t w7, int16_t w8,
    const int16_t *A, int16_t *B);

// BoxFilter — uniform-weight K×K filter. Single-channel signed 8/16-bit on
// PVA via libpva_operator's pvaBoxFilter{Create,Submit}. No coefficient
// tensor (the filter is implicitly 1/K² everywhere). REPLICATE border.
// Output saturates to dtype range. M/N are full image dims; the shim
// writes a (M-2)×(N-2) interior to caller-supplied B starting at &B[1][1]
// (same pointer-shift convention the matcher uses for conv2d).
void polygeist_pva_boxfilter_3x3_i8(int32_t M, int32_t N,
                                     const int8_t *A, int8_t *B);
void polygeist_pva_boxfilter_3x3_i16(int32_t M, int32_t N,
                                      const int16_t *A, int16_t *B);

// GaussianFilter — separable Gaussian via PVA's pvaGaussianFilter. The
// hardware takes (sigmaX, sigmaY, kernelSize) parameters; for the v0
// integration we hardcode kernelSize=3 and sigmaX=sigmaY=1.0 (the natural
// 3×3 Gaussian). Surfacing sigma as launch operands is future work; the
// matcher would need to recognize Gaussian-weighted convs and route here
// instead of to OpConv2d.
void polygeist_pva_gaussian_3x3_i8(int32_t M, int32_t N,
                                    const int8_t *A, int8_t *B);
void polygeist_pva_gaussian_3x3_i16(int32_t M, int32_t N,
                                     const int16_t *A, int16_t *B);

// BilateralFilter — edge-preserving smoothing. PVA's pvaBilateralFilter
// hardcodes sigmaRange=25.0 / sigmaSpace=10.0 (typical edge-preserving
// parameters) for v0. CPU stub is approximate (matches PVA within a few
// LSBs on typical-content images; bilateral is non-linear so bit-exact
// match is impractical to model without the full PVA fixed-point spec).
// Validation strategy: PVA must run cleanly + output must be in-range.
void polygeist_pva_bilateral_3x3_i8(int32_t M, int32_t N,
                                     const int8_t *A, int8_t *B);
void polygeist_pva_bilateral_3x3_i16(int32_t M, int32_t N,
                                      const int16_t *A, int16_t *B);

// HistogramEqualization — U8-only on PVA; we reinterpret i8 bytes as u8
// (bitwise identical) for the shim's tensor allocation.
void polygeist_pva_histeq_i8(int32_t M, int32_t N,
                              const int8_t *A, int8_t *B);

// ============================================================================
// Extracted-darknet batched CNN-block primitives. All four take 4D NCHW
// tensors (and 1D per-channel vectors for batchnorm) as raw FP32 pointers
// plus the shape parameters. The CUDA backend wires each to its
// corresponding cuDNN forward call; the CPU stub runs a reference loop
// for correctness validation.
//
// These cover every primitive in a ResNet residual block except ReLU:
//   conv + bn + (relu) + conv + bn + add.
// ============================================================================

// Batched multi-channel 2D convolution (forward, NCHW, FP32):
//   Out[b,oc,oh,ow] = sum_{ic,kh,kw} A[b,ic,oh+kh,ow+kw] * F[oc,ic,kh,kw]
// No padding, stride 1, no dilation, no activation. K is the (square)
// filter size, OH = H - K + 1, OW = W - K + 1.
void polygeist_cudnn_conv2d_batched(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F, float *Out);

// Darknet-style explicit im2col + GEMM fused to one convolution. Single
// batch, NCHW, FP32. Supports caller-supplied square kernel, stride, and pad.
void polygeist_cudnn_conv2d_im2col_gemm_f32(
    int32_t IC, int32_t H, int32_t W, int32_t OC,
    int32_t K, int32_t S, int32_t P,
    const float *A, const float *F, float *Out);

// Batched multi-channel 2D max pooling (forward, NCHW, FP32).
// Window size K and stride S are derived from H/OH (assumed K == stride
// for the common ResNet shapes; tweak the shim if needed). OH and OW are
// the output spatial dims after pooling.
void polygeist_cudnn_maxpool_batched(
    int32_t B, int32_t C, int32_t H, int32_t W, int32_t OH, int32_t OW,
    const float *A, float *Out);

// Batched per-channel batch normalization (INFERENCE mode, NCHW, FP32):
//   Out[b,c,h,w] = scale[c] * (A[b,c,h,w] - mean[c]) * inv_std[c] + bias[c]
// where inv_std[c] = 1/sqrt(var[c] + eps) is pre-computed by the caller.
// The CUDA backend uses cudnnBatchNormalizationForwardInference (which
// expects mean + variance, not inv_std). The shim recovers variance via
//   var = 1/inv_std² - eps_assumed (eps_assumed = 1e-5).
// This is an inversion of the kernel's pre-baked inv_std; the caller
// must use the same eps when building inv_std for bit-exact output.
void polygeist_cudnn_batchnorm_inference(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out);

// Batched 4D elementwise tensor add (ResNet residual shortcut, FP32):
//   Out[b,c,h,w] += A[b,c,h,w]
// The CUDA backend uses cudnnAddTensor with α=β=1.
void polygeist_cudnn_add_tensor_batched(
    int32_t B, int32_t C, int32_t H, int32_t W,
    const float *A, float *Out);

// 1×1 conv via batched gemm. Mathematically:
//   C[b, oc, h, w] = sum_ic A[b, ic, h, w] * F[oc, ic, 0, 0]
//
// Since NCHW packs IC-contiguous H*W planes, A[b] is naturally a 2D
// matrix of shape (IC, H*W) (row-major). Per batch:
//   C[b] (OC, H*W) = F (OC, IC) × A[b] (IC, H*W)
// → cublasSgemmStridedBatched with batchCount=B, F shared (stride 0),
// A and C strided by IC*H*W and OC*H*W respectively. Hits tensor cores
// on Orin for IC, OC, H*W aligned to 8.
//
// The signature takes M = B*H*W (flattened parallel dims), N = OC,
// K = IC. The harness/lowering passes B*H*W as M; the shim recovers
// B and H*W via the assumption that A is contiguous NCHW (which the
// row-major layout guarantees for a single 1×1 conv).
void polygeist_cublas_sgemm_1x1conv(
    int32_t B, int32_t IC, int32_t OC, int32_t HW,
    const float *A, const float *F, float *C);

// Symmetric rank-K update — AᵀA or A·Aᵀ. FP32, row-major.
//   C[N,N] = Aᵀ·A   where A is K×N (so AᵀA is N×N, symmetric)
// Only the upper triangle of C is computed; the lower is mirrored on
// host before returning so the caller can treat C as fully populated.
// Routes to cublasSsyrk_v2 — half the flops of the equivalent gemm.
void polygeist_cublas_dsyrk(
    int32_t N, int32_t K, const float *A, float *C);

// Fused matmul + bias + relu, FP32. Computes:
//   C[m,n] = relu(sum_k A[m,k] * B[k,n] + bias[n])
// A is MxK, B is KxN, C is MxN, bias is length N (broadcast over rows).
// Routes to cublasLt's CUBLASLT_EPILOGUE_RELU_BIAS — needs -lcublasLt at link.
void polygeist_cublaslt_matmul_bias_relu(
    int32_t M, int32_t N, int32_t K,
    const float *A, const float *B, const float *bias,
    float *C);

// Fused conv + bias + residual-add + relu, FP32 NCHW. Computes:
//   Out[b,oc,oh,ow] = relu(conv(A,F)[b,oc,oh,ow] + bias[oc] + Z[b,oc,oh,ow])
//
// Bias is per-output-channel (length OC); Z has the same shape as Out
// and is the ResNet skip-connection input. The CUDA backend issues one
// cudnnConvolutionBiasActivationForward with α₁=1, α₂=1, activation=RELU.
void polygeist_cudnn_conv_bias_relu_add_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *bias, const float *Z,
    float *Out);

// Fused conv + bn (inference) + relu, FP32 NCHW. Computes:
//   Out[b,oc,oh,ow] = relu(
//       scale[oc] * (conv(A, F)[b,oc,oh,ow] - mean[oc]) * inv_std[oc]
//       + bias[oc])
//
// This is the canonical ResNet inner pattern. The CUDA backend uses the
// standard BN-folding trick — pre-compute a scaled filter and an
// effective bias on the host, then issue a single
// cudnnConvolutionBiasActivationForward call with CUDNN_ACTIVATION_RELU.
// Folded filter / bias are:
//   F'[oc,ic,kh,kw] = F[oc,ic,kh,kw] * scale[oc] * inv_std[oc]
//   b'[oc]         = bias[oc] - scale[oc] * mean[oc] * inv_std[oc]
// With those substitutions, conv + bn-inference + relu = act(conv(F') + b'),
// which cudnnConvolutionBiasActivationForward computes natively in one
// kernel — the bandwidth-bound bn and relu ride the compute-bound conv
// instead of paying their own per-call setup.
void polygeist_cudnn_conv_bn_relu_fused(
    int32_t B, int32_t IC, int32_t OC,
    int32_t H, int32_t W, int32_t K,
    const float *A, const float *F,
    const float *scale, const float *mean,
    const float *inv_std, const float *bias,
    float *Out);

// llama2.c RMSNorm, FP32:
//   Out[i] = Weight[i] * X[i] * rsqrt(sum_j X[j]^2 / N + 1e-5)
void polygeist_rmsnorm_f32(
    int32_t N, const float *X, const float *Weight, float *Out);
void polygeist_rmsnorm_unweighted_f32(
    int32_t N, const float *X, float *Out);
void polygeist_cublas_dot_f32(
    int32_t N, const float *X, const float *Y, float *Out);
void polygeist_cuda_gelu_tanh_f32(
    int32_t N, const float *X, float *Out);
void polygeist_whisper_exp_shift_sum_f32(
    int32_t N, const float *X, float max_val, float *Out, float *Sum);

// llama2.c row softmax, FP32, in-place:
//   X[i] = exp(X[i] - max(X)) / sum_j exp(X[j] - max(X))
// CUDA backend routes this through cudnnSoftmaxForward.
void polygeist_cudnn_softmax_forward_f32(int32_t N, float *X);
void polygeist_cudnn_softmax_forward_out_f32(
    int32_t N, const float *X, float *Out);

// Llama standalone FP32 helpers. The CUDA backend implements these with
// CUDA-runtime copies plus cuBLAS/cuDNN tensor ops; the CPU backend is a
// reference implementation for host correctness runs.
void polygeist_cuda_copy_f32(int32_t N, const float *X, float *Out);
void polygeist_cuda_add_f32(
    int32_t N, const float *X, const float *Y, float *Out);
void polygeist_cuda_mask_select_f32(
    int32_t N, int32_t pos, const float *Scores, float *Out);
void polygeist_cuda_swiglu_f32(
    int32_t N, const float *Gate, const float *Up, float *Out);
void polygeist_cuda_rope_mulmul_f32(
    int32_t M, int32_t N, const float *A, const float *B,
    const float *C, const float *D, float *Out, int32_t add);

// Per-call CUDA-event timing (CUDA backend only — CPU stub returns 0.0).
// Pair with polygeist_cublas_time_begin / polygeist_cublas_time_end around
// a sequence of kernel calls.
void  polygeist_cublas_time_begin(void);
double polygeist_cublas_time_end_ms(void);  // returns ms since last begin

#ifdef __cplusplus
}
#endif

#endif  // POLYGEIST_CUBLAS_RT_H
