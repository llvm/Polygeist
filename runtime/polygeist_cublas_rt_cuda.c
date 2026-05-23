// polygeist_cublas_rt_cuda.c — real cuBLAS implementation of the runtime
// shim ABI. Compile with nvcc (or clang+CUDA) and link against -lcublas
// -lcudart. Build with:
//   nvcc -O3 -c polygeist_cublas_rt_cuda.c -o polygeist_cublas_rt.o
// or, treating the file as C with the cuda toolkit headers in scope:
//   clang -O3 -I${CUDA}/include -c polygeist_cublas_rt_cuda.c -o ...
//
// MEMORY MODEL (initial, per-op copies):
//   For each polygeist_cublas_dgemm call we cudaMalloc A_dev / B_dev / C_dev,
//   cudaMemcpy H→D, run cublasDgemm, cudaMemcpy D→H, cudaFree. This is
//   correct but slow: copies dominate for small matrices. The follow-up
//   work is a "device-residency analysis" pass that hoists allocs to the
//   enclosing function entry and elides intermediate copies between
//   consecutive launches.
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

  double *dA = NULL, *dB = NULL, *dC = NULL;
  CUDA_CHECK(cudaMalloc((void**)&dA, bytes_A));
  CUDA_CHECK(cudaMalloc((void**)&dB, bytes_B));
  CUDA_CHECK(cudaMalloc((void**)&dC, bytes_C));

  CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_A, cudaMemcpyHostToDevice, g_stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, B, bytes_B, cudaMemcpyHostToDevice, g_stream));
  if (beta != 0.0) {
    CUDA_CHECK(cudaMemcpyAsync(dC, C, bytes_C, cudaMemcpyHostToDevice, g_stream));
  }

  // Row-major C = α A·B + β C   computed in column-major as
  //   Cᵀ = α Bᵀ·Aᵀ + β Cᵀ
  // i.e. cublasDgemm(handle, N_op, N_op, n=N, m=M, k=K, &α, B, ldb, A, lda, &β, C, ldc).
  CUBLAS_CHECK(cublasDgemm(g_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            /*m=*/N, /*n=*/M, /*k=*/K,
                            &alpha,
                            dB, ldb,
                            dA, lda,
                            &beta,
                            dC, ldc));

  CUDA_CHECK(cudaMemcpyAsync(C, dC, bytes_C, cudaMemcpyDeviceToHost, g_stream));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
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

// cuDNN 9-tap conv2d (PolyBench filter hardcoded). Single-image,
// single-channel, FP64, 3x3 no-padding stride-1.
void polygeist_cudnn_conv2d_polybench9tap(
    int32_t M, int32_t N, const double *A, double *B) {
  polygeist_cublas_init();
  ensure_cudnn();

  // PolyBench's 3x3 weight matrix (matches kernel_conv2d in
  // third_party/polybenchGpu/OpenMP/stencils/convolution-2d/).
  static const double filter_h[9] = {
     0.2,  0.5, -0.8,
    -0.3,  0.6, -0.9,
     0.4,  0.7,  0.1,
  };

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
