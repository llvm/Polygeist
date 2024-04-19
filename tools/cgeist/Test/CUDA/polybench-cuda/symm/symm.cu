// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * symm.c: This file is part of the PolyBench/C 3.2 test suite.
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

static void init_array(int ni, int nj, double *C, double *A, double *B,
                       double *tmp) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      C[i * nj + j] = ((double)i * j) / ni;
      B[i * nj + j] = ((double)i * j) / ni;
      tmp[i * nj + j] = 0;
    }
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++)
      A[i * nj + j] = ((double)i * j) / ni;
}

static void print_array(int ni, int nj, double *C) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, "%0.2lf ", C[i * nj + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_tmp(int m, int n, double alpha, double beta, double *C,
                           double *A, double *B, double *tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n) {
    tmp[i * n + j] = 0;
    for (int k = 0; k < i; k++)
      tmp[i * n + j] += B[k * n + j] * A[i * n + k];
  }
}

__global__ void kernel_C(int m, int n, double alpha, double beta, double *C,
                         double *A, double *B, double *tmp) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n)
    C[i * n + j] = beta * C[i * n + j] + alpha * B[i * n + j] * A[i * n + i] +
                   alpha * tmp[i * n + j];
}

__global__ void kernel_sum(int m, int n, double alpha, double beta, double *C,
                           double *A, double *B, double *tmp) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (k < m - 1 && j < n) {
    for (int i = k + 1; i < m; i++)
      C[k * n + j] += alpha * B[i * n + j] * A[i * n + k];
  }
}

static void kernel(int m, int n, double alpha, double beta, double *C,
                   double *A, double *B, double *tmp) {
  const unsigned int threadsPerBlock = 256;

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m, block.x), num_blocks(n, block.y), 1);
    kernel_tmp<<<grid, block>>>(m, n, alpha, beta, C, A, B, tmp);
  }

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m, block.x), num_blocks(n, block.y), 1);
    kernel_C<<<grid, block>>>(m, n, alpha, beta, C, A, B, tmp);
  }

  // TODO: Combine both kernels?
  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m - 1, block.x), num_blocks(n, block.y), 1);
    kernel_sum<<<grid, block>>>(m, n, alpha, beta, C, A, B, tmp);
  }
}

int main(int argc, char **argv) {
  int dump_code = atoi(argv[1]);
  int ni = atoi(argv[2]);
  int nj = atoi(argv[3]);

  double *A = (double *)malloc(nj * nj * sizeof(double));
  double *B = (double *)malloc(ni * nj * sizeof(double));
  double *C = (double *)malloc(ni * nj * sizeof(double));
  double *tmp = (double *)malloc(ni * nj * sizeof(double));

  double alpha = 32412;
  double beta = 2123;

  init_array(ni, nj, C, A, B, tmp);

  double *dev_A;
  double *dev_B;
  double *dev_C;
  double *dev_tmp;
  cudaMalloc(&dev_A, nj * nj * sizeof(double));
  cudaMalloc(&dev_B, ni * nj * sizeof(double));
  cudaMalloc(&dev_C, ni * nj * sizeof(double));
  cudaMalloc(&dev_tmp, ni * nj * sizeof(double));
  cudaMemcpy(dev_A, A, nj * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, ni * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, ni * nj * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_tmp, tmp, ni * nj * sizeof(double), cudaMemcpyHostToDevice);

  kernel(ni, nj, alpha, beta, dev_C, dev_A, dev_B, dev_tmp);
  cudaMemcpy(C, dev_C, ni * nj * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(ni, nj, C);

  free((void *)C);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;

  return 0;
}
