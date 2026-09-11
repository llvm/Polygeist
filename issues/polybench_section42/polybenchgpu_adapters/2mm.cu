// Canonical LARGE/FP64 adapter around the verbatim PolyBenchGPU 2MM kernels
// from commit 5584aaa7, CUDA/2MM/2mm.cu.
#include "common.cuh"

#define NI 800
#define NJ 900
#define NK 1100
#define NL 1200
#define _PB_NI ni
#define _PB_NJ nj
#define _PB_NK nk
#define _PB_NL nl
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, double alpha,
                            double beta, double *tmp, double *A, double *B) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_NI) && (j < _PB_NJ)) {
    tmp[i * NJ + j] = 0;
    int k;
    for (k = 0; k < _PB_NK; k++)
      tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
  }
}

__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, double alpha,
                            double beta, double *tmp, double *C, double *D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_NI) && (j < _PB_NL)) {
    D[i * NL + j] *= beta;
    int k;
    for (k = 0; k < _PB_NJ; k++)
      D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
  }
}

extern "C" void kernel_2mm(int ni, int nj, int nk, int nl, double alpha,
                           double beta, double tmp[NI][NJ], double A[NI][NK],
                           double B[NK][NJ], double C[NJ][NL],
                           double D[NI][NL]) {
  double *dt, *da, *db, *dc, *dd;
  pbgpu_checked(cudaMalloc(&dt, sizeof(double) * NI * NJ), "malloc tmp");
  pbgpu_checked(cudaMalloc(&da, sizeof(double) * NI * NK), "malloc A");
  pbgpu_checked(cudaMalloc(&db, sizeof(double) * NK * NJ), "malloc B");
  pbgpu_checked(cudaMalloc(&dc, sizeof(double) * NJ * NL), "malloc C");
  pbgpu_checked(cudaMalloc(&dd, sizeof(double) * NI * NL), "malloc D");
  pbgpu_checked(
      cudaMemcpy(da, A, sizeof(double) * NI * NK, cudaMemcpyHostToDevice),
      "copy A");
  pbgpu_checked(
      cudaMemcpy(db, B, sizeof(double) * NK * NJ, cudaMemcpyHostToDevice),
      "copy B");
  pbgpu_checked(
      cudaMemcpy(dc, C, sizeof(double) * NJ * NL, cudaMemcpyHostToDevice),
      "copy C");
  pbgpu_checked(
      cudaMemcpy(dd, D, sizeof(double) * NI * NL, cudaMemcpyHostToDevice),
      "copy D");
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid1((NJ + block.x - 1) / block.x, (NI + block.y - 1) / block.y);
  dim3 grid2((NL + block.x - 1) / block.x, (NI + block.y - 1) / block.y);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start event");
  pbgpu_checked(cudaEventCreate(&stop), "create stop event");
  pbgpu_checked(cudaEventRecord(start), "record start event");
  mm2_kernel1<<<grid1, block>>>(ni, nj, nk, nl, alpha, beta, dt, da, db);
  mm2_kernel2<<<grid2, block>>>(ni, nj, nk, nl, alpha, beta, dt, dc, dd);
  pbgpu_checked(cudaGetLastError(), "launch 2MM kernels");
  pbgpu_report(start, stop);
  pbgpu_checked(
      cudaMemcpy(D, dd, sizeof(double) * NI * NL, cudaMemcpyDeviceToHost),
      "copy D");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dt);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
}
