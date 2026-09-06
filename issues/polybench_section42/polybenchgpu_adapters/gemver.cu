// Canonical LARGE/FP64 adapter around the verbatim PolyBenchGPU GEMVER
// kernels from commit 5584aaa7, CUDA/GEMVER/gemver.cu.
#include "common.cuh"
#define N 2000
#define _PB_N n

__global__ void gemver_kernel1(int n, double alpha, double beta, double *a,
                               double *v1, double *v2, double *u1, double *u2) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_N) && (j < _PB_N))
    a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
}
__global__ void gemver_kernel2(int n, double alpha, double beta, double *a,
                               double *x, double *y, double *z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < _PB_N) {
    int j;
    for (j = 0; j < _PB_N; j++)
      x[i] += beta * a[j * N + i] * y[j];
    x[i] += z[i];
  }
}
__global__ void gemver_kernel3(int n, double alpha, double beta, double *a,
                               double *x, double *w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= 0) && (i < _PB_N)) {
    int j;
    for (j = 0; j < _PB_N; j++)
      w[i] += alpha * a[i * N + j] * x[j];
  }
}

extern "C" void kernel_gemver(int n, double alpha, double beta, double A[N][N],
                              double u1[N], double v1[N], double u2[N],
                              double v2[N], double w[N], double x[N],
                              double y[N], double z[N]) {
  double *da, *du1, *dv1, *du2, *dv2, *dw, *dx, *dy, *dz;
  pbgpu_checked(cudaMalloc(&da, sizeof(double) * N * N), "malloc A");
  pbgpu_checked(cudaMalloc(&du1, sizeof(double) * N), "malloc u1");
  pbgpu_checked(cudaMalloc(&dv1, sizeof(double) * N), "malloc v1");
  pbgpu_checked(cudaMalloc(&du2, sizeof(double) * N), "malloc u2");
  pbgpu_checked(cudaMalloc(&dv2, sizeof(double) * N), "malloc v2");
  pbgpu_checked(cudaMalloc(&dw, sizeof(double) * N), "malloc w");
  pbgpu_checked(cudaMalloc(&dx, sizeof(double) * N), "malloc x");
  pbgpu_checked(cudaMalloc(&dy, sizeof(double) * N), "malloc y");
  pbgpu_checked(cudaMalloc(&dz, sizeof(double) * N), "malloc z");
  pbgpu_checked(
      cudaMemcpy(da, A, sizeof(double) * N * N, cudaMemcpyHostToDevice),
      "copy A");
  pbgpu_checked(cudaMemcpy(du1, u1, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy u1");
  pbgpu_checked(cudaMemcpy(dv1, v1, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy v1");
  pbgpu_checked(cudaMemcpy(du2, u2, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy u2");
  pbgpu_checked(cudaMemcpy(dv2, v2, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy v2");
  pbgpu_checked(cudaMemcpy(dw, w, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy w");
  pbgpu_checked(cudaMemcpy(dx, x, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy x");
  pbgpu_checked(cudaMemcpy(dy, y, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy y");
  pbgpu_checked(cudaMemcpy(dz, z, sizeof(double) * N, cudaMemcpyHostToDevice),
                "copy z");
  dim3 b1(32, 8), g1((N + 31) / 32, (N + 7) / 8);
  dim3 b2(256, 1), g2((N + 255) / 256, 1);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start event");
  pbgpu_checked(cudaEventCreate(&stop), "create stop event");
  pbgpu_checked(cudaEventRecord(start), "record start event");
  gemver_kernel1<<<g1, b1>>>(n, alpha, beta, da, dv1, dv2, du1, du2);
  gemver_kernel2<<<g2, b2>>>(n, alpha, beta, da, dx, dy, dz);
  gemver_kernel3<<<g2, b2>>>(n, alpha, beta, da, dx, dw);
  pbgpu_checked(cudaGetLastError(), "launch GEMVER kernels");
  pbgpu_report(start, stop);
  pbgpu_checked(
      cudaMemcpy(A, da, sizeof(double) * N * N, cudaMemcpyDeviceToHost),
      "copy A");
  pbgpu_checked(cudaMemcpy(x, dx, sizeof(double) * N, cudaMemcpyDeviceToHost),
                "copy x");
  pbgpu_checked(cudaMemcpy(w, dw, sizeof(double) * N, cudaMemcpyDeviceToHost),
                "copy w");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(da);
  cudaFree(du1);
  cudaFree(dv1);
  cudaFree(du2);
  cudaFree(dv2);
  cudaFree(dw);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz);
}
