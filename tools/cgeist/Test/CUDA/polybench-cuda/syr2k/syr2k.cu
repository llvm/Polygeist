// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * syr2k.c: This file is part of the PolyBench/C 3.2 test suite.
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

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_beta(int n, int m, double alpha, double beta, double *C,
                            double *A, double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j <= i)
    C[i * n + j] *= beta;
}

__global__ void kernel_product(int n, int m, double alpha, double beta,
                               double *C, double *A, double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < n && j <= i) {
    for (int k = 0; k < m; k++)
      C[i * n + j] += A[j * m + k] * alpha * B[i * m + k] +
                      B[j * m + k] * alpha * A[i * m + k];
  }
}

static void kernel(int n, int m, double alpha, double beta, double *C,
                   double *A, double *B) {
  const unsigned int threadsPerBlock = 256;

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(n, block.y), 1);
    kernel_beta<<<grid, block>>>(n, m, alpha, beta, C, A, B);
  }

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(n, block.y), 1);
    kernel_product<<<grid, block>>>(n, m, alpha, beta, C, A, B);
  }
}

static void init_array(int ni, int nj, double *C, double *A, double *B) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      A[i * nj + j] = ((double)i * j) / ni;
      B[i * nj + j] = ((double)i * j) / ni;
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++)
      C[i * ni + j] = ((double)i * j) / ni;
}

static void print_array(int ni, double *C) {
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) {
      fprintf(stderr, "%0.2lf ", C[i * ni + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
  int dump_code = atoi(argv[1]);
  int m = atoi(argv[2]);
  int n = atoi(argv[3]);

  double *A = (double *)malloc(m * n * sizeof(double));
  double *B = (double *)malloc(m * n * sizeof(double));
  double *C = (double *)malloc(m * m * sizeof(double));

  double alpha = 32412;
  double beta = 2123;

  init_array(m, n, C, A, B);

  double *dev_A;
  double *dev_B;
  double *dev_C;
  cudaMalloc(&dev_A, n * m * sizeof(double));
  cudaMalloc(&dev_B, n * m * sizeof(double));
  cudaMalloc(&dev_C, n * m * sizeof(double));
  cudaMemcpy(dev_A, A, n * m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, n * m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C, C, n * m * sizeof(double), cudaMemcpyHostToDevice);

  kernel(m, n, alpha, beta, dev_C, dev_A, dev_B);
  cudaMemcpy(C, dev_C, n * m * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(m, C);

  free((void *)C);
  ;
  free((void *)A);
  ;
  free((void *)B);
  ;

  return 0;
}
