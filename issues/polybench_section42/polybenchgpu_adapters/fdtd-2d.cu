#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define N 1
#define TMAX 500
#define NX 1000
#define NY 1200
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_fdtd_2d
#include "CUDA/FDTD-2D/fdtd2d.cu"
#undef main

extern "C" void kernel_fdtd_2d(int tmax, int nx, int ny,
                                double ex[NX][NY], double ey[NX][NY],
                                double hz[NX][NY], double fict[TMAX]) {
  double *df, *dex, *dey, *dhz;
  size_t bytes = sizeof(double) * NX * NY;
  pbgpu_checked(cudaMalloc(&df, sizeof(double) * TMAX), "malloc fict");
  pbgpu_checked(cudaMalloc(&dex, bytes), "malloc ex");
  pbgpu_checked(cudaMalloc(&dey, bytes), "malloc ey");
  pbgpu_checked(cudaMalloc(&dhz, bytes), "malloc hz");
  pbgpu_checked(cudaMemcpy(df, fict, sizeof(double) * TMAX,
                           cudaMemcpyHostToDevice), "copy fict");
  pbgpu_checked(cudaMemcpy(dex, ex, bytes, cudaMemcpyHostToDevice), "copy ex");
  pbgpu_checked(cudaMemcpy(dey, ey, bytes, cudaMemcpyHostToDevice), "copy ey");
  pbgpu_checked(cudaMemcpy(dhz, hz, bytes, cudaMemcpyHostToDevice), "copy hz");
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((NY + block.x - 1) / block.x, (NX + block.y - 1) / block.y);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start");
  pbgpu_checked(cudaEventCreate(&stop), "create stop");
  pbgpu_checked(cudaEventRecord(start), "record start");
  for (int t = 0; t < tmax; ++t) {
    fdtd_step1_kernel<<<grid, block>>>(nx, ny, df, dex, dey, dhz, t);
    fdtd_step2_kernel<<<grid, block>>>(nx, ny, dex, dey, dhz, t);
    fdtd_step3_kernel<<<grid, block>>>(nx, ny, dex, dey, dhz, t);
  }
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(ex, dex, bytes, cudaMemcpyDeviceToHost), "copy ex");
  pbgpu_checked(cudaMemcpy(ey, dey, bytes, cudaMemcpyDeviceToHost), "copy ey");
  pbgpu_checked(cudaMemcpy(hz, dhz, bytes, cudaMemcpyDeviceToHost), "copy hz");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(df); cudaFree(dex); cudaFree(dey); cudaFree(dhz);
}
