// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 5
// clang-format on
/**
 * cholesky.c: This file is part of the PolyBench 3.0 test suite.
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

/* Array initialization. */
static void init_array(int n, double *A) {
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      A[i * n + j] = 1.0 / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double *A)

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, "%0.2lf ", A[i * n + j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
}

__global__ void kernel0(int n, int j, double *A) {
  A[j * n + j] = std::sqrt(A[j * n + j]);
}

__global__ void kernel1(int n, int j, double *A) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n && i > j)
    A[i * n + j] /= A[j * n + j];
}

__global__ void kernel2(int n, int j, double *A) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;

  if (j < n && j < i && i < n && j < k && k <= i)
    A[i * n + k] -= A[i * n + j] * A[k * n + j];
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

static void kernel_polly(int n, double *dev_A) {
  const unsigned int threadsPerBlock = 256;

  for (int iter = 0; iter < n; iter++) {
    kernel0<<<1, 1>>>(n, iter, dev_A);

    kernel1<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, iter,
                                                                 dev_A);

    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(n, block.x), num_blocks(n, block.y), 1);
    kernel2<<<block, grid>>>(n, iter, dev_A);
  }
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = atoi(argv[2]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(n * n * sizeof(double));

  /* Initialize array(s). */
  init_array(n, A);

  double *dev_A;
  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);

  kernel_polly(n, dev_A);

  cudaMemcpy(A, dev_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(n, A);

  /* Be clean. */
  free((void *)A);

  return 0;
}
