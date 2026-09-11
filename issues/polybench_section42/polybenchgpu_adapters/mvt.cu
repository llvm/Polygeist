#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define N 2000
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_mvt
#include "CUDA/MVT/mvt.cu"
#undef main

extern "C" void kernel_mvt(int n, double x1[N], double x2[N], double y1[N],
                            double y2[N], double A[N][N]) {
  double *dA, *dx1, *dx2, *dy1, *dy2;
  pbgpu_checked(cudaMalloc(&dA, sizeof(double) * N * N), "malloc A");
  pbgpu_checked(cudaMalloc(&dx1, sizeof(double) * N), "malloc x1");
  pbgpu_checked(cudaMalloc(&dx2, sizeof(double) * N), "malloc x2");
  pbgpu_checked(cudaMalloc(&dy1, sizeof(double) * N), "malloc y1");
  pbgpu_checked(cudaMalloc(&dy2, sizeof(double) * N), "malloc y2");
  cudaMemcpy(dA, A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dx1, x1, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dx2, x2, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dy1, y1, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dy2, y2, sizeof(double) * N, cudaMemcpyHostToDevice);
  dim3 block(256, 1), grid((N + 255) / 256);
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  mvt_kernel1<<<grid, block>>>(n, dA, dx1, dy1);
  mvt_kernel2<<<grid, block>>>(n, dA, dx2, dy2);
  pbgpu_report(start, stop);
  cudaMemcpy(x1, dx1, sizeof(double) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(x2, dx2, sizeof(double) * N, cudaMemcpyDeviceToHost);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(dx1); cudaFree(dx2); cudaFree(dy1); cudaFree(dy2);
}
