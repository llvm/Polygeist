// PolyBenchGPU commit 5584aaa7 DOITGEN kernels, normalized to the canonical
// PolyBench/C LARGE FP64 dimensions. The computational loop bodies are
// unchanged; only types, dimensions, ABI glue, and timing are normalized.
#include "common.cuh"
#include <cstring>

#define NR 150
#define NQ 140
#define NP 160

__global__ void pbgpu_doitgen_kernel1(double *sum, const double *A,
                                       const double *C4, int r) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (p < NP && q < NQ) {
    sum[r * (NQ * NP) + q * NP + p] = 0.0;
    for (int s = 0; s < NP; ++s)
      sum[r * (NQ * NP) + q * NP + p] +=
          A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
  }
}

__global__ void pbgpu_doitgen_kernel2(const double *sum, double *A, int r) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (p < NP && q < NQ)
    A[r * (NQ * NP) + q * NP + p] =
        sum[r * (NQ * NP) + q * NP + p];
}

extern "C" void kernel_doitgen(int nr, int nq, int np,
                                double A[NR][NQ][NP], double C4[NP][NP],
                                double sum[NP]) {
  (void)nr; (void)nq; (void)np;
  double *dA, *dC4, *dSum;
  size_t a_bytes = sizeof(double) * NR * NQ * NP;
  pbgpu_checked(cudaMalloc(&dA, a_bytes), "malloc A");
  pbgpu_checked(cudaMalloc(&dC4, sizeof(double) * NP * NP), "malloc C4");
  pbgpu_checked(cudaMalloc(&dSum, a_bytes), "malloc sum");
  pbgpu_checked(cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice), "copy A");
  pbgpu_checked(cudaMemcpy(dC4, C4, sizeof(double) * NP * NP,
                           cudaMemcpyHostToDevice), "copy C4");
  dim3 block(32, 8), grid((NP + 31) / 32, (NQ + 7) / 8);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  for (int r = 0; r < NR; ++r) {
    pbgpu_doitgen_kernel1<<<grid, block>>>(dSum, dA, dC4, r);
    pbgpu_doitgen_kernel2<<<grid, block>>>(dSum, dA, r);
  }
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(A, dA, a_bytes, cudaMemcpyDeviceToHost), "copy A");
  memcpy(sum, &A[NR - 1][NQ - 1][0], sizeof(double) * NP);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dC4); cudaFree(dSum);
}
