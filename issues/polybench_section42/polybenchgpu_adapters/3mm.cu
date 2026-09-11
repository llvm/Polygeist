// Canonical LARGE/FP64 adapter around the verbatim PolyBenchGPU 3MM kernels
// from commit 5584aaa7, CUDA/3MM/3mm.cu.
#include "common.cuh"

#define NI 800
#define NJ 900
#define NK 1000
#define NL 1100
#define NM 1200
#define _PB_NI ni
#define _PB_NJ nj
#define _PB_NK nk
#define _PB_NL nl
#define _PB_NM nm
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, double *A,
                            double *B, double *E) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_NI) && (j < _PB_NJ)) {
    E[i * NJ + j] = 0;
    int k;
    for (k = 0; k < _PB_NK; k++)
      E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
  }
}
__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, double *C,
                            double *D, double *F) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_NJ) && (j < _PB_NL)) {
    F[i * NL + j] = 0;
    int k;
    for (k = 0; k < _PB_NM; k++)
      F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
  }
}
__global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, double *E,
                            double *F, double *G) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < _PB_NI) && (j < _PB_NL)) {
    G[i * NL + j] = 0;
    int k;
    for (k = 0; k < _PB_NJ; k++)
      G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
  }
}

extern "C" void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
                           double E[NI][NJ], double A[NI][NK], double B[NK][NJ],
                           double F[NJ][NL], double C[NJ][NM], double D[NM][NL],
                           double G[NI][NL]) {
  double *da, *db, *dc, *dd, *de, *df, *dg;
  pbgpu_checked(cudaMalloc(&da, sizeof(double) * NI * NK), "malloc A");
  pbgpu_checked(cudaMalloc(&db, sizeof(double) * NK * NJ), "malloc B");
  pbgpu_checked(cudaMalloc(&dc, sizeof(double) * NJ * NM), "malloc C");
  pbgpu_checked(cudaMalloc(&dd, sizeof(double) * NM * NL), "malloc D");
  pbgpu_checked(cudaMalloc(&de, sizeof(double) * NI * NJ), "malloc E");
  pbgpu_checked(cudaMalloc(&df, sizeof(double) * NJ * NL), "malloc F");
  pbgpu_checked(cudaMalloc(&dg, sizeof(double) * NI * NL), "malloc G");
  pbgpu_checked(
      cudaMemcpy(da, A, sizeof(double) * NI * NK, cudaMemcpyHostToDevice),
      "copy A");
  pbgpu_checked(
      cudaMemcpy(db, B, sizeof(double) * NK * NJ, cudaMemcpyHostToDevice),
      "copy B");
  pbgpu_checked(
      cudaMemcpy(dc, C, sizeof(double) * NJ * NM, cudaMemcpyHostToDevice),
      "copy C");
  pbgpu_checked(
      cudaMemcpy(dd, D, sizeof(double) * NM * NL, cudaMemcpyHostToDevice),
      "copy D");
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 g1((NJ + 31) / 32, (NI + 7) / 8), g2((NL + 31) / 32, (NJ + 7) / 8),
      g3((NL + 31) / 32, (NI + 7) / 8);
  cudaEvent_t start, stop;
  pbgpu_checked(cudaEventCreate(&start), "create start event");
  pbgpu_checked(cudaEventCreate(&stop), "create stop event");
  pbgpu_checked(cudaEventRecord(start), "record start event");
  mm3_kernel1<<<g1, block>>>(ni, nj, nk, nl, nm, da, db, de);
  mm3_kernel2<<<g2, block>>>(ni, nj, nk, nl, nm, dc, dd, df);
  mm3_kernel3<<<g3, block>>>(ni, nj, nk, nl, nm, de, df, dg);
  pbgpu_checked(cudaGetLastError(), "launch 3MM kernels");
  pbgpu_report(start, stop);
  pbgpu_checked(
      cudaMemcpy(G, dg, sizeof(double) * NI * NL, cudaMemcpyDeviceToHost),
      "copy G");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
  cudaFree(de);
  cudaFree(df);
  cudaFree(dg);
}
