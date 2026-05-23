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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cublasHandle_t g_handle;
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
