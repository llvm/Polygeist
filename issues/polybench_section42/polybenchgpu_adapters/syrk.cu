// PolyBenchGPU SYRK full-matrix kernel at canonical LARGE/FP64 dimensions.
// Device timing includes the upstream computational kernel's full N x N work.
#include "common.cuh"
#include <cstdlib>

#define N 1200
#define M 1000

// Verbatim computational body from PolyBenchGPU commit 5584aaa7, normalized
// only for names, FP64, and canonical compile-time dimensions.
__global__ void pbgpu_syrk_kernel(int n, int m, double alpha, double beta,
                                  const double *a, double *c) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    c[i * N + j] *= beta;
    for (int k = 0; k < m; ++k)
      c[i * N + j] += alpha * a[i * M + k] * a[j * M + k];
  }
}

extern "C" void kernel_syrk(int n, int m, double alpha, double beta,
                             double C[N][N], double A[N][M]) {
  double *dA, *dC;
  double *result = static_cast<double *>(std::malloc(sizeof(double) * N * N));
  pbgpu_checked(cudaMalloc(&dA, sizeof(double) * N * M), "malloc A");
  pbgpu_checked(cudaMalloc(&dC, sizeof(double) * N * N), "malloc C");
  pbgpu_checked(cudaMemcpy(dA, A, sizeof(double) * N * M,
                           cudaMemcpyHostToDevice), "copy A");
  pbgpu_checked(cudaMemcpy(dC, C, sizeof(double) * N * N,
                           cudaMemcpyHostToDevice), "copy C");
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  dim3 block(32, 8), grid((N + 31) / 32, (N + 7) / 8);
  pbgpu_syrk_kernel<<<grid, block>>>(n, m, alpha, beta, dA, dC);
  pbgpu_checked(cudaGetLastError(), "launch SYRK");
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(result, dC, sizeof(double) * N * N,
                           cudaMemcpyDeviceToHost), "copy C back");
  // The external kernel computes the full matrix. Return the canonical live
  // lower triangle so that those computed values receive the normal numeric
  // correctness check; the extra upper-triangle work remains in the timing.
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      C[i][j] = result[(size_t)i * N + j];
  std::free(result);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dC);
}
