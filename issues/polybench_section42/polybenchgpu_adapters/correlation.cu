#include <cuda.h>
#include "common.cuh"
#include "upstream_namespace.cuh"
#define M 1200
#define N 1400
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define main polybenchgpu_upstream_main_correlation
#include "CUDA/CORR/correlation.cu"
#undef main

__global__ void pbgpu_corr_mean(int m, int n, double float_n, double *mean,
                                 const double *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m) {
    mean[j] = 0.0;
    for (int i = 0; i < n; ++i) mean[j] += data[i * M + j];
    mean[j] /= float_n;
  }
}
__global__ void pbgpu_corr_std(int m, int n, double float_n,
                               const double *mean, double *stddev,
                               const double *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m) {
    stddev[j] = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = data[i * M + j] - mean[j];
      stddev[j] += d * d;
    }
    stddev[j] = sqrt(stddev[j] / float_n);
    if (stddev[j] <= 0.1) stddev[j] = 1.0;
  }
}
__global__ void pbgpu_corr_reduce(int m, int n, double float_n,
                                  const double *mean, const double *stddev,
                                  double *data) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m)
    data[i * M + j] = (data[i * M + j] - mean[j]) /
                      (sqrt(float_n) * stddev[j]);
}
__global__ void pbgpu_corr_matrix(int m, int n, double *corr,
                                  const double *data) {
  int j1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (j1 < m) {
    corr[j1 * M + j1] = 1.0;
    for (int j2 = j1 + 1; j2 < m; ++j2) {
      double value = 0.0;
      for (int i = 0; i < n; ++i)
        value += data[i * M + j1] * data[i * M + j2];
      corr[j1 * M + j2] = value;
      corr[j2 * M + j1] = value;
    }
  }
}

extern "C" void kernel_correlation(int m, int n, double float_n,
                                    double data[N][M], double corr[M][M],
                                    double mean[M], double stddev[M]) {
  double *ddata, *dcorr, *dmean, *dstd;
  size_t data_bytes = sizeof(double) * N * M;
  size_t corr_bytes = sizeof(double) * M * M;
  pbgpu_checked(cudaMalloc(&ddata, data_bytes), "malloc data");
  pbgpu_checked(cudaMalloc(&dcorr, corr_bytes), "malloc corr");
  pbgpu_checked(cudaMalloc(&dmean, sizeof(double) * M), "malloc mean");
  pbgpu_checked(cudaMalloc(&dstd, sizeof(double) * M), "malloc stddev");
  pbgpu_checked(cudaMemcpy(ddata, data, data_bytes, cudaMemcpyHostToDevice),
                 "copy data");
  dim3 one(256), two(32, 8);
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  pbgpu_corr_mean<<<(M + 255) / 256, one>>>(m, n, float_n, dmean, ddata);
  pbgpu_corr_std<<<(M + 255) / 256, one>>>(m, n, float_n, dmean, dstd, ddata);
  pbgpu_corr_reduce<<<dim3((M + 31) / 32, (N + 7) / 8), two>>>(
      m, n, float_n, dmean, dstd, ddata);
  pbgpu_corr_matrix<<<(M + 255) / 256, one>>>(m, n, dcorr, ddata);
  pbgpu_report(start, stop);
  pbgpu_checked(cudaMemcpy(corr, dcorr, corr_bytes, cudaMemcpyDeviceToHost),
                 "copy corr");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(ddata); cudaFree(dcorr); cudaFree(dmean); cudaFree(dstd);
}
