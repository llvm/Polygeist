// Canonical PolyBench/C harness adapter for the unmodified PolyBenchGPU GEMM
// computational kernel. The included external source is pinned in the run
// manifest; only allocation, transfer, timing, and ABI glue is implemented
// here. gemm_kernel itself remains the upstream implementation.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define NI 1000
#define NJ 1100
#define NK 1200
#define DATA_TYPE double
#define _PB_NI ni
#define _PB_NJ nj
#define _PB_NK nk
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

// Verbatim computational kernel from PolyBenchGPU commit 5584aaa7,
// CUDA/GEMM/gemm.cu. Only whitespace is normalized.
__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha,
                            DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b,
                            DATA_TYPE *c) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i < _PB_NI) && (j < _PB_NJ)) {
    c[i * NJ + j] *= beta;
    int k;
    for (k = 0; k < _PB_NK; k++) {
      c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ + j];
    }
  }
}

static void checked(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "native PolyBenchGPU %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::abort();
  }
}

extern "C" void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                            double C[NI][NJ], double A[NI][NK],
                            double B[NK][NJ]) {
  double *a = nullptr;
  double *b = nullptr;
  double *c = nullptr;
  checked(cudaMalloc(&a, sizeof(double) * NI * NK), "cudaMalloc(A)");
  checked(cudaMalloc(&b, sizeof(double) * NK * NJ), "cudaMalloc(B)");
  checked(cudaMalloc(&c, sizeof(double) * NI * NJ), "cudaMalloc(C)");
  checked(cudaMemcpy(a, A, sizeof(double) * NI * NK, cudaMemcpyHostToDevice),
          "copy A to device");
  checked(cudaMemcpy(b, B, sizeof(double) * NK * NJ, cudaMemcpyHostToDevice),
          "copy B to device");
  checked(cudaMemcpy(c, C, sizeof(double) * NI * NJ, cudaMemcpyHostToDevice),
          "copy C to device");

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((NJ + block.x - 1) / block.x, (NI + block.y - 1) / block.y);
  cudaEvent_t start;
  cudaEvent_t stop;
  checked(cudaEventCreate(&start), "create start event");
  checked(cudaEventCreate(&stop), "create stop event");
  checked(cudaEventRecord(start), "record start event");
  gemm_kernel<<<grid, block>>>(ni, nj, nk, alpha, beta, a, b, c);
  checked(cudaGetLastError(), "launch gemm_kernel");
  checked(cudaEventRecord(stop), "record stop event");
  checked(cudaEventSynchronize(stop), "synchronize stop event");
  float device_ms = 0.0f;
  checked(cudaEventElapsedTime(&device_ms, start, stop), "measure device time");
  std::fprintf(stderr, "POLYBENCH_NATIVE_GPU_TIMING device_ms=%.9f\n",
               static_cast<double>(device_ms));

  checked(cudaMemcpy(C, c, sizeof(double) * NI * NJ, cudaMemcpyDeviceToHost),
          "copy C to host");
  checked(cudaEventDestroy(start), "destroy start event");
  checked(cudaEventDestroy(stop), "destroy stop event");
  checked(cudaFree(a), "cudaFree(A)");
  checked(cudaFree(b), "cudaFree(B)");
  checked(cudaFree(c), "cudaFree(C)");
}
