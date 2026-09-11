#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define N 1200
#define NI 1000
#define NJ 1200
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_gramschmidt
#include "CUDA/GRAMSCHM/gramschmidt.cu"
#undef main

extern "C" void kernel_gramschmidt(int m, int n, double A[NI][NJ],
                                    double R[NJ][NJ], double Q[NI][NJ]) {
  double *dA, *dR, *dQ;
  size_t a_bytes = sizeof(double) * NI * NJ;
  size_t r_bytes = sizeof(double) * NJ * NJ;
  pbgpu_checked(cudaMalloc(&dA, a_bytes), "malloc A");
  pbgpu_checked(cudaMalloc(&dR, r_bytes), "malloc R");
  pbgpu_checked(cudaMalloc(&dQ, a_bytes), "malloc Q");
  pbgpu_checked(cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice), "copy A");
  // These kernels index only threadIdx.x. A one-dimensional block avoids
  // the upstream launch's duplicate threadIdx.x values from blockDim.y=8.
  dim3 block(256, 1);
  dim3 one(1), rows((NI + block.x - 1) / block.x);
  dim3 cols((NJ + block.x - 1) / block.x);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  for (int k = 0; k < n; ++k) {
    gramschmidt_kernel1<<<one, block>>>(m, n, dA, dR, dQ, k);
    gramschmidt_kernel2<<<rows, block>>>(m, n, dA, dR, dQ, k);
    gramschmidt_kernel3<<<cols, block>>>(m, n, dA, dR, dQ, k);
  }
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(R, dR, r_bytes, cudaMemcpyDeviceToHost), "copy R");
  pbgpu_checked(cudaMemcpy(Q, dQ, a_bytes, cudaMemcpyDeviceToHost), "copy Q");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dR); cudaFree(dQ);
}
