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
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Intentionally do NOT include <cuda_fp16.h> or <cuda_bf16.h>. Those
 * headers use NVCC-specific `__device__` builtins that fail to parse under
 * aarch64-linux-gnu-gcc (our cross-compile path). cuDNN's API is type-agnostic
 * on the data side — it reads the buffer layout from the descriptor
 * (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16 / etc.), so we use uint16_t* for
 * the device buffers in the half-precision paths instead of __half /
 * __nv_bfloat16. Bits are identical, so memcpy from the host's _Float16 /
 * __bf16 arrays via uint16_t lands the correct values on the device. */

static cublasHandle_t g_handle;
static cudnnHandle_t  g_cudnn = NULL;
static cudaStream_t   g_stream;
static cudaEvent_t    g_ev_begin;
static cudaEvent_t    g_ev_end;
static int            g_initialized = 0;

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

static void ensure_cudnn(void) {
  if (g_cudnn) return;
  CUDNN_CHECK(cudnnCreate(&g_cudnn));
  CUDNN_CHECK(cudnnSetStream(g_cudnn, g_stream));
}

// Zero-copy helper: pin a host buffer for direct GPU access on Jetson's
// unified memory. Silently tolerates re-registration of the same pointer
// (e.g. when A and B alias for syrk-shape calls). Returns the device-side
// pointer obtained via cudaHostGetDevicePointer (equals the host pointer
// under UVA on Tegra, but the explicit translation is safer).
//
// We tried bypassing cudaHostRegister and passing host pointers directly
// to cuBLAS — fails with illegal-memory-access. cuBLAS requires the
// buffer to be registered (or device-allocated) even on a Tegra SoC
// where the iGPU can technically reach any DRAM page.
static void *register_host_safe(void *ptr, size_t bytes) {
  cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterMapped);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    fprintf(stderr, "%s:%d cudaHostRegister(%p, %zu) failed: %s\n",
            __FILE__, __LINE__, ptr, bytes, cudaGetErrorString(err));
    abort();
  }
  void *dev = NULL;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev, ptr, 0));
  return dev;
}

static void unregister_host_safe(void *ptr) {
  cudaError_t err = cudaHostUnregister(ptr);
  if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
    fprintf(stderr, "%s:%d cudaHostUnregister(%p) failed: %s\n",
            __FILE__, __LINE__, ptr, cudaGetErrorString(err));
  }
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

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_B = (size_t)K * (size_t)ldb * sizeof(double);
  size_t bytes_C = (size_t)M * (size_t)ldc * sizeof(double);

  // Pin host buffers for direct GPU access (zero-copy on Jetson).
  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dB = (double *)register_host_safe((void *)B, bytes_B);
  double *dC = (double *)register_host_safe(C, bytes_C);

  // Row-major C = α A·B + β C  →  col-major Cᵀ = α Bᵀ·Aᵀ + β Cᵀ
  CUBLAS_CHECK(cublasDgemm(g_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M, /*k=*/K,
                            &alpha,
                            dB, ldb,
                            dA, lda,
                            &beta,
                            dC, ldc));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)B);
  unregister_host_safe(C);
}

// Host-side memset. In the current no-hoisting model the array lives on
// host between launches; pulling it to device just to zero is wasteful.
void polygeist_cublas_memset_zero_2d(int32_t M, int32_t N,
                                       double *A, int32_t lda) {
  if (lda == N) {
    // Contiguous: one memset.
    memset(A, 0, (size_t)M * (size_t)N * sizeof(double));
  } else {
    for (int32_t i = 0; i < M; ++i) {
      memset(&A[(size_t)i * (size_t)lda], 0,
             (size_t)N * sizeof(double));
    }
  }
}

// y = α*x + β*y (axpby). O(N) bandwidth-bound; H↔D copy + two cuBLAS
// calls would dominate any GPU benefit. Do it on the host directly.
void polygeist_cublas_daxpby(int32_t N, double alpha, const double *x,
                              double beta, double *y) {
  for (int32_t i = 0; i < N; ++i) y[i] = alpha * x[i] + beta * y[i];
}

// y += x (axpy with α=1).
void polygeist_cublas_daxpy_unit(int32_t N, const double *x, double *y) {
  polygeist_cublas_init();
  size_t bytes = (size_t)N * sizeof(double);
  double *dx = (double *)register_host_safe((void *)x, bytes);
  double *dy = (double *)register_host_safe(y, bytes);
  double one = 1.0;
  CUBLAS_CHECK(cublasDaxpy(g_handle, N, &one, dx, 1, dy, 1));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

// Rank-2 update: A += u1·v1ᵀ + u2·v2ᵀ (gemver body). Two cublasDger calls.
void polygeist_cublas_dger_rank2(int32_t M, int32_t N,
                                   const double *u1, const double *v1,
                                   const double *u2, const double *v2,
                                   double *A, int32_t lda) {
  polygeist_cublas_init();
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
  CUBLAS_CHECK(cublasDger(g_handle, /*m=*/N, /*n=*/M,
                          &one, dv1, 1, du1, 1, dA, lda));
  CUBLAS_CHECK(cublasDger(g_handle, /*m=*/N, /*n=*/M,
                          &one, dv2, 1, du2, 1, dA, lda));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  unregister_host_safe(A);
  unregister_host_safe((void *)u1);
  unregister_host_safe((void *)v1);
  unregister_host_safe((void *)u2);
  unregister_host_safe((void *)v2);
}

// Host-side 1D memset. Same justification as the 2D variant — host copy
// to device just to zero is wasteful.
void polygeist_cublas_memset_zero_1d(int32_t N, double *v) {
  memset(v, 0, (size_t)N * sizeof(double));
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

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_x = (size_t)N * sizeof(double);
  size_t bytes_y = (size_t)M * sizeof(double);

  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dx = (double *)register_host_safe((void *)x, bytes_x);
  double *dy = (double *)register_host_safe(y, bytes_y);

  // Row-major y = A·x  →  col-major view of A is Aᵀ; OP_T undoes that.
  CUBLAS_CHECK(cublasDgemv(g_handle,
                            CUBLAS_OP_T,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

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

  size_t bytes_A = (size_t)M * (size_t)lda * sizeof(double);
  size_t bytes_x = (size_t)M * sizeof(double);   // x is M for Aᵀ·x
  size_t bytes_y = (size_t)N * sizeof(double);   // y is N for Aᵀ·x

  double *dA = (double *)register_host_safe((void *)A, bytes_A);
  double *dx = (double *)register_host_safe((void *)x, bytes_x);
  double *dy = (double *)register_host_safe(y, bytes_y);

  CUBLAS_CHECK(cublasDgemv(g_handle,
                            CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M,
                            &alpha,
                            dA, lda,
                            dx, 1,
                            &beta,
                            dy, 1));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  unregister_host_safe((void *)A);
  unregister_host_safe((void *)x);
  unregister_host_safe(y);
}

// Host-side scale. Could use cublasDscal but the H↔D copy overhead would
// dominate this O(MN) op; do it on the CPU side. Future device-residency
// hoisting will make this a GPU op.
void polygeist_cublas_dscal_2d(int32_t M, int32_t N, double scale,
                                 double *A, int32_t lda) {
  for (int32_t i = 0; i < M; ++i) {
    double *row = &A[(size_t)i * (size_t)lda];
    for (int32_t j = 0; j < N; ++j) row[j] *= scale;
  }
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
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));

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
  CUDNN_CHECK(cudnnConvolutionForward(
      g_cudnn, &alpha, in_desc, dA, f_desc, dF, conv_desc,
      algo_perf.algo, dWS, ws_size, &beta, out_desc, dB));

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
