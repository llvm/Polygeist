// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 5
// clang-format on
/**
 * jacobi-2d-imper.c: This file is part of the PolyBench/C 3.2 test suite.
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

__global__ void kernel_stencil(int n, double *A, double *B) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

  if (i < n - 1 && j < n - 1) {
    B[i * n + j] = (A[i * n + j] + A[i * n + j - 1] + A[i * n + 1 + j] +
                    A[(1 + i) * n + j] + A[(i - 1) * n + j]) /
                   5;
  }
}

static void kernel(int tsteps, int n, double *A, double *B) {
  const unsigned int threadsPerBlock = 256;

  for (int t = 1; t <= tsteps; t++) {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n - 2, block.x), num_blocks(n - 2, block.y), 1);
    kernel_stencil<<<grid, block>>>(n, A, B);
    kernel_stencil<<<grid, block>>>(n, B, A);
  }
}

static void init_array(int n, double *A, double *B) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i * n + j] = ((double)i * (j + 2) + 2) / n;
      B[i * n + j] = ((double)i * (j + 3) + 3) / n;
    }
}

static void print_array(int n, double *A)

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, "%0.2lf ", A[i * n + j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int n = atoi(argv[2]);
  int tsteps = atoi(argv[3]);

  double *A = (double *)malloc(n * n * sizeof(double));
  double *B = (double *)malloc(n * n * sizeof(double));

  // __builtin_assume(tsteps>0);
  // __builtin_assume(n>2);
  init_array(n, A, B);

  double *dev_A;
  double *dev_B;
  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMalloc(&dev_B, n * n * sizeof(double));
  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

  kernel(tsteps, n, dev_A, dev_B);
  cudaMemcpy(A, dev_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(n, A);

  free((void *)A);
  ;
  free((void *)B);
  ;
  cudaFree((void *)dev_A);
  cudaFree((void *)dev_B);

  return 0;
}
