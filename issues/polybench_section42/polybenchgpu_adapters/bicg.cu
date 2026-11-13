#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define N 2100
#define NX 2100
#define NY 1900
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_bicg
#include "CUDA/BICG/bicg.cu"
#undef main

extern "C" void kernel_bicg(int m, int n, double A[NX][NY], double s[NY],
                             double q[NX], double p[NY], double r[NX]) {
  double *dA, *ds, *dq, *dp, *dr;
  pbgpu_checked(cudaMalloc(&dA, sizeof(double) * NX * NY), "malloc A");
  pbgpu_checked(cudaMalloc(&ds, sizeof(double) * NY), "malloc s");
  pbgpu_checked(cudaMalloc(&dq, sizeof(double) * NX), "malloc q");
  pbgpu_checked(cudaMalloc(&dp, sizeof(double) * NY), "malloc p");
  pbgpu_checked(cudaMalloc(&dr, sizeof(double) * NX), "malloc r");
  pbgpu_checked(cudaMemcpy(dA, A, sizeof(double) * NX * NY,
                           cudaMemcpyHostToDevice), "copy A");
  pbgpu_checked(cudaMemcpy(dp, p, sizeof(double) * NY,
                           cudaMemcpyHostToDevice), "copy p");
  pbgpu_checked(cudaMemcpy(dr, r, sizeof(double) * NX,
                           cudaMemcpyHostToDevice), "copy r");
  dim3 block(256, 1), grid_s((NY + 255) / 256), grid_q((NX + 255) / 256);
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  bicg_kernel1<<<grid_s, block>>>(n, m, dA, dr, ds);
  bicg_kernel2<<<grid_q, block>>>(n, m, dA, dp, dq);
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(s, ds, sizeof(double) * NY,
                           cudaMemcpyDeviceToHost), "copy s");
  pbgpu_checked(cudaMemcpy(q, dq, sizeof(double) * NX,
                           cudaMemcpyDeviceToHost), "copy q");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(dA); cudaFree(ds); cudaFree(dq); cudaFree(dp); cudaFree(dr);
}
