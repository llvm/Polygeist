// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * correlation.c: This file is part of the PolyBench/C 3.2 test suite.
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

#define EPS 0.1

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_mean(int m, int n, double *data, double *corr,
                            double mean[], double stddev[]) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < m) {
    mean[j] = 0.0;
    for (int i = 0; i < n; i++)
      mean[j] += data[i * m + j];
    mean[j] /= n;
  }
}

__global__ void kernel_stddev(int m, int n, double *data, double *corr,
                              double mean[], double stddev[]) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < m) {
    stddev[j] = 0.0;
    for (int i = 0; i < n; i++)
      stddev[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
    stddev[j] /= n;
    stddev[j] = sqrt(stddev[j]);
    /* The following in an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero-
       divide. */
    if (stddev[j] <= EPS)
      stddev[j] = 1.0;
  }
}

__global__ void kernel_reduce(int m, int n, double *data, double *corr,
                              double mean[], double stddev[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j < m) {
    data[i * m + j] -= mean[j];
    data[i * m + j] /= std::sqrt((double)n) * stddev[j];
  }
}

__global__ void kernel_diag(int m, int n, double *data, double *corr,
                            double mean[], double stddev[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m) {
    corr[i * m + i] = 1.0;
  }
}

__global__ void kernel_corr(int m, int n, double *data, double *corr,
                            double mean[], double stddev[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y + i + 1;

  if (i < m - 1 && j < m) {
    corr[i * m + j] = 0.0;
    for (int k = 0; k < n; k++)
      corr[i * m + j] += (data[k * m + i] * data[k * m + j]);
    corr[j * m + i] = corr[i * m + j];
  }
}

__global__ void kernel_tail(int m, int n, double *data, double *corr,
                            double mean[], double stddev[]) {
  corr[(m - 1) * m + m - 1] = 1.0;
}

static void kernel(int m, int n, double *data, double *corr, double mean[],
                   double stddev[]) {
  const unsigned threadsPerBlock = 256;

  kernel_mean<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(
      m, n, data, corr, mean, stddev);
  kernel_stddev<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(
      m, n, data, corr, mean, stddev);

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(m, block.y), 1);
    kernel_reduce<<<block, grid>>>(m, n, data, corr, mean, stddev);
  }

  kernel_diag<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(
      m, n, data, corr, mean, stddev);

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m - 1, block.x), num_blocks(m - 1, block.y), 1);
    kernel_corr<<<block, grid>>>(m, n, data, corr, mean, stddev);
  }

  kernel_tail<<<1, 1>>>(m, n, data, corr, mean, stddev);
}

static void init_array(int m, int n, double *data) {
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i * n + j] = ((double)i * j) / 1000;
}

static void print_array(int m, double *corr)

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf(stderr, "%0.2lf ", corr[i * m + j]);
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
  double *stddev = (double *)malloc(m * sizeof(double));
  double *corr = (double *)malloc(m * m * sizeof(double));

  init_array(m, n, data);

  double *dev_data;
  double *dev_mean;
  double *dev_stddev;
  double *dev_corr;
  cudaMalloc(&dev_data, n * m * sizeof(double));
  cudaMalloc(&dev_mean, m * sizeof(double));
  cudaMalloc(&dev_stddev, m * sizeof(double));
  cudaMalloc(&dev_corr, m * m * sizeof(double));
  cudaMemcpy(dev_data, data, n * m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mean, mean, m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_stddev, stddev, m * m * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_corr, corr, m * m * sizeof(double), cudaMemcpyHostToDevice);

  kernel(m, n, dev_data, dev_corr, dev_mean, dev_stddev);
  cudaMemcpy(corr, dev_corr, m * m * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(m, corr);

  free((void *)data);
  ;
  free((void *)corr);
  ;
  free((void *)mean);
  ;
  free((void *)stddev);
  ;

  return 0;
}
