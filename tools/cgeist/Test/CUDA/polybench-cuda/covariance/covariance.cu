// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * covariance.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void init_array(int m, int n, double *data) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i * m + j] = ((double)i * j) / 1000;
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_mean(int m, int n, double data[], double cov[],
                            double mean[]) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < m) {
    mean[j] = 0.0;
    for (int i = 0; i < n; i++)
      mean[j] += data[i * m + j];
    mean[j] /= n;
  }
}

__global__ void kernel_reduce(int m, int n, double data[], double cov[],
                              double mean[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j < m) {
    data[i * m + j] -= mean[j];
  }
}

__global__ void kernel_cov(int m, int n, double data[], double cov[],
                           double mean[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + i;

  if (i < m && j < m) {
    cov[i * m + j] = 0.0;
    for (int k = 0; k < n; k++)
      cov[i * m + j] += data[k * m + i] * data[k * m + j];
    cov[i * m + j] /= (n - 1.0);
    cov[j * m + i] = cov[i * m + j];
  }
}

static void kernel(int m, int n, double data[], double cov[], double mean[]) {
  const unsigned threadsPerBlock = 256;

  kernel_mean<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(m, n, data,
                                                                   cov, mean);

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(m, block.y), 1);
    kernel_reduce<<<block, grid>>>(m, n, data, cov, mean);
  }

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m - 1, block.x), num_blocks(m - 1, block.y), 1);
    kernel_cov<<<block, grid>>>(m, n, data, cov, mean);
  }
}

static void print_array(int m, double *cov)

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf(stderr, "%0.2lf ", cov[i * m + j]);
      if ((i * m + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int n = atoi(argv[2]);
  int m = atoi(argv[3]);

  double *data = (double *)malloc(n * m * sizeof(double));
  double *mean = (double *)malloc(m * sizeof(double));
  double *cov = (double *)malloc(m * m * sizeof(double));

  init_array(m, n, data);

  double *dev_data;
  double *dev_mean;
  double *dev_cov;
  cudaMalloc(&dev_data, n * m * sizeof(double));
  cudaMalloc(&dev_mean, m * sizeof(double));
  cudaMalloc(&dev_cov, m * m * sizeof(double));
  cudaMemcpy(dev_data, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mean, mean, m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_cov, cov, m * m * sizeof(double), cudaMemcpyHostToDevice);

  kernel(m, n, dev_data, dev_cov, dev_mean);
  cudaMemcpy(cov, dev_cov, m * m * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(m, cov);

  free((void *)data);
  free((void *)cov);
  free((void *)mean);

  return 0;
}
