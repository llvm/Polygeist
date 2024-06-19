// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 100
// RUN: cgeist %s %stdinclude %cudaopts_polymer -O3 -o %s.execm && %s.execm 1 100
// clang-format on
/**
 * lu.c: This file is part of the PolyBench/C 3.2 test suite.
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

static void init_array(int n, double *A) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i * n + j] = ((double)(i + 1) * (j + 1)) / n;
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

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_div(int n, double *A, int k) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + k + 1;

  if (i < n)
    A[i * n + k] /= A[k * n + k];
}

__global__ void kernel_A(int n, double *A, int k) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + k + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y + k + 1;

  if (i < n && j < n)
    A[i * n + j] -= A[i * n + k] * A[k * n + j];
}

static void kernel(int n, double *A) {
  const unsigned int threadsPerBlock = 256;

  for (int iter = 0; iter < n - 1; iter++) {
    kernel_div<<<num_blocks(n - (iter + 1), threadsPerBlock),
                 threadsPerBlock>>>(n, A, iter);

    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n - (iter + 1), block.x),
              num_blocks(n - (iter + 1), block.y), 1);
    kernel_A<<<grid, block>>>(n, A, iter);
  }
}

int main(int argc, char **argv) {

  int dump_code = atoi(argv[1]);
  int n = atoi(argv[2]);

  double *A = (double *)malloc(n * n * sizeof(double));

  init_array(n, A);

  double *dev_A;
  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);

  kernel(n, dev_A);

  cudaMemcpy(A, dev_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  if (dump_code == 1)
    print_array(n, A);

  free((void *)A);
  ;

  return 0;
}
