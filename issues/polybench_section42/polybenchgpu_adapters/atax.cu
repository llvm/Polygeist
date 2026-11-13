#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define N 2100
#define NX 1900
#define NY 2100
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_atax
#include "CUDA/ATAX/atax.cu"
#undef main

extern "C" void kernel_atax(int m, int n, double A[NX][NY], double x[NY],
                             double y[NY], double tmp[NX]) {
  double *dA, *dx, *dy, *dtmp;
  pbgpu_checked(cudaMalloc(&dA, sizeof(double) * NX * NY), "malloc A");
  pbgpu_checked(cudaMalloc(&dx, sizeof(double) * NY), "malloc x");
  pbgpu_checked(cudaMalloc(&dy, sizeof(double) * NY), "malloc y");
  pbgpu_checked(cudaMalloc(&dtmp, sizeof(double) * NX), "malloc tmp");
  pbgpu_checked(cudaMemcpy(dA, A, sizeof(double) * NX * NY,
                           cudaMemcpyHostToDevice), "copy A");
  pbgpu_checked(cudaMemcpy(dx, x, sizeof(double) * NY,
                           cudaMemcpyHostToDevice), "copy x");
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((NX + block.x - 1) / block.x);
  dim3 grid2((NY + block.x - 1) / block.x);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  atax_kernel1<<<grid1, block>>>(m, n, dA, dx, dtmp);
  atax_kernel2<<<grid2, block>>>(m, n, dA, dy, dtmp);
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(y, dy, sizeof(double) * NY,
                           cudaMemcpyDeviceToHost), "copy y");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dtmp);
}
