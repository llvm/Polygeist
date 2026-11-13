// Canonical LARGE/FP64 adapter around the verbatim PolyBenchGPU GESUMMV
// kernel from commit 5584aaa7, CUDA/GESUMMV/gesummv.cu.
#include "common.cuh"
#define N 1300
#define _PB_N n
#define DIM_THREAD_BLOCK_X 256

__global__ void gesummv_kernel(int n, double alpha, double beta, double *A,
                               double *B, double *tmp, double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < _PB_N) {
    int j;
    for (j = 0; j < _PB_N; j++) {
      tmp[i] += A[i * N + j] * x[j];
      y[i] += B[i * N + j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}

extern "C" void kernel_gesummv(int n, double alpha, double beta, double A[N][N],
                               double B[N][N], double tmp[N], double x[N],
                               double y[N]) {
  double *da, *db, *dt, *dx, *dy;
  pbgpu_checked(cudaMalloc(&da, sizeof(double) * N * N), "malloc A");
  pbgpu_checked(cudaMalloc(&db, sizeof(double) * N * N), "malloc B");
  pbgpu_checked(cudaMalloc(&dt, sizeof(double) * N), "malloc tmp");
  pbgpu_checked(cudaMalloc(&dx, sizeof(double) * N), "malloc x");
  pbgpu_checked(cudaMalloc(&dy, sizeof(double) * N), "malloc y");
  pbgpu_checked(
      cudaMemcpy(da, A, sizeof(double) * N * N, cudaMemcpyHostToDevice),
      "copy A");
  pbgpu_checked(
      cudaMemcpy(db, B, sizeof(double) * N * N, cudaMemcpyHostToDevice),
      "copy B");
  pbgpu_checked(cudaMemcpy(dx, x, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy x");
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start event");
  pbgpu_checked(cudaEventCreate(&stop), "create stop event");
  pbgpu_checked(cudaEventRecord(start), "record start event");
  pbgpu_checked(cudaMemset(dt, 0, sizeof(double) * N), "zero tmp");
  pbgpu_checked(cudaMemset(dy, 0, sizeof(double) * N), "zero y");
  gesummv_kernel<<<(N + DIM_THREAD_BLOCK_X - 1) / DIM_THREAD_BLOCK_X,
                   DIM_THREAD_BLOCK_X>>>(n, alpha, beta, da, db, dt, dx, dy);
  pbgpu_checked(cudaGetLastError(), "launch gesummv kernel");
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(tmp, dt, sizeof(double) * N, cudaMemcpyDeviceToHost),
                "copy tmp");
  pbgpu_checked(cudaMemcpy(y, dy, sizeof(double) * N, cudaMemcpyDeviceToHost),
                "copy y");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dt);
  cudaFree(dx);
  cudaFree(dy);
}
