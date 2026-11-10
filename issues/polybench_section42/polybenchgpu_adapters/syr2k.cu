// PolyBenchGPU SYR2K full-matrix kernel at canonical LARGE/FP64 dimensions.
#include "common.cuh"
#include <cstdlib>

#define N 1200
#define M 1000

// Verbatim computational body from PolyBenchGPU commit 5584aaa7, normalized
// only for names, FP64, and canonical compile-time dimensions.
__global__ void pbgpu_syr2k_kernel(int n, int m, double alpha, double beta,
                                   const double *a, const double *b,
                                   double *c) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    c[i * N + j] *= beta;
    for (int k = 0; k < m; ++k)
      c[i * N + j] += alpha * a[i * M + k] * b[j * M + k] +
                      alpha * b[i * M + k] * a[j * M + k];
  }
}

extern "C" void kernel_syr2k(int n, int m, double alpha, double beta,
                              double C[N][N], double A[N][M],
                              double B[N][M]) {
  double *dA, *dB, *dC;
  double *result = static_cast<double *>(std::malloc(sizeof(double) * N * N));
  pbgpu_checked(cudaMalloc(&dA, sizeof(double) * N * M), "malloc A");
  pbgpu_checked(cudaMalloc(&dB, sizeof(double) * N * M), "malloc B");
  pbgpu_checked(cudaMalloc(&dC, sizeof(double) * N * N), "malloc C");
  pbgpu_checked(cudaMemcpy(dA, A, sizeof(double) * N * M,
                           cudaMemcpyHostToDevice), "copy A");
  pbgpu_checked(cudaMemcpy(dB, B, sizeof(double) * N * M,
                           cudaMemcpyHostToDevice), "copy B");
  pbgpu_checked(cudaMemcpy(dC, C, sizeof(double) * N * N,
                           cudaMemcpyHostToDevice), "copy C");
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  dim3 block(32, 8), grid((N + 31) / 32, (N + 7) / 8);
  pbgpu_syr2k_kernel<<<grid, block>>>(n, m, alpha, beta, dA, dB, dC);
  pbgpu_checked(cudaGetLastError(), "launch SYR2K");
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(result, dC, sizeof(double) * N * N,
                           cudaMemcpyDeviceToHost), "copy C back");
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      C[i][j] = result[(size_t)i * N + j];
  std::free(result);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
