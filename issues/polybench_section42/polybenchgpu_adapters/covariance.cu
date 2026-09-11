#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define M 1200
#define N 1400
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_covariance
#include "CUDA/COVAR/covariance.cu"
#undef main

__global__ void pbgpu_covar_mean(int m, int n, double float_n, double *mean,
                                  const double *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m) {
    mean[j] = 0.0;
    for (int i = 0; i < n; ++i) mean[j] += data[i * M + j];
    mean[j] /= float_n;
  }
}
__global__ void pbgpu_covar_reduce(int m, int n, const double *mean,
                                   double *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) data[i * M + j] -= mean[j];
}
__global__ void pbgpu_covar_matrix(int m, int n, double float_n,
                                   double *cov, const double *data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m)
    for (int j = i; j < m; ++j) {
      double value = 0.0;
      for (int k = 0; k < n; ++k)
        value += data[k * M + i] * data[k * M + j];
      value /= float_n - 1.0;
      cov[i * M + j] = value;
      cov[j * M + i] = value;
    }
}

extern "C" void kernel_covariance(int m, int n, double float_n,
                                   double data[N][M], double cov[M][M],
                                   double mean[M]) {
  double *ddata, *dcov, *dmean;
  size_t data_bytes = sizeof(double) * N * M;
  size_t cov_bytes = sizeof(double) * M * M;
  pbgpu_checked(cudaMalloc(&ddata, data_bytes), "malloc data");
  pbgpu_checked(cudaMalloc(&dcov, cov_bytes), "malloc cov");
  pbgpu_checked(cudaMalloc(&dmean, sizeof(double) * M), "malloc mean");
  pbgpu_checked(cudaMemcpy(ddata, data, data_bytes, cudaMemcpyHostToDevice),
                 "copy data");
  dim3 one(256), two(32, 8);
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  pbgpu_covar_mean<<<(M + 255) / 256, one>>>(m, n, float_n, dmean, ddata);
  pbgpu_covar_reduce<<<dim3((M + 31) / 32, (N + 7) / 8), two>>>(
      m, n, dmean, ddata);
  pbgpu_covar_matrix<<<(M + 255) / 256, one>>>(m, n, float_n, dcov, ddata);
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(cov, dcov, cov_bytes, cudaMemcpyDeviceToHost),
                 "copy cov");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(ddata); cudaFree(dcov); cudaFree(dmean);
}
