// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10 10
// clang-format on
/**
 * trmm.c: This file is part of the PolyBench 3.0 test suite.
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
static void init_array(int n, int m, double *alpha, double *A, double *B) {
  int i, j;

  *alpha = 32412;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i * m + j] = ((double)i * j) / m;
      B[j * n + j] = ((double)i * j) / n;
    }
}

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_contract(int n, int m, double alpha, double *B,
                                double *A) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < n) {
    for (int i = 0; i < m; i++)
      for (int k = i + 1; k < m; k++)
        B[i * n + j] += A[k * m + i] * B[k * n + j];
  }
}

__global__ void kernel_alpha(int n, int m, double alpha, double *B, double *A) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n)
    B[i * n + j] *= alpha;
}

static void kernel(int n, int m, double alpha, double *B, double *A) {
  const unsigned int threadsPerBlock = 256;

  kernel_contract<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
      n, m, alpha, B, A);

  {
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(m, block.x), num_blocks(n, block.y), 1);
    kernel_alpha<<<grid, block>>>(n, m, alpha, B, A);
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n, double *B) {
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, "%0.2lf ", B[i * n + j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = atoi(argv[2]);
  int m = atoi(argv[3]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double alpha;
  double *A = (double *)malloc(n * m * sizeof(double));
  double *B = (double *)malloc(m * n * sizeof(double));

  /* Initialize array(s). */
  init_array(n, m, &alpha, A, B);

  double *dev_A;
  double *dev_B;
  cudaMalloc(&dev_A, n * m * sizeof(double));
  cudaMalloc(&dev_B, m * n * sizeof(double));
  cudaMemcpy(dev_A, A, n * m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, m * n * sizeof(double), cudaMemcpyHostToDevice);
  /* Run kernel. */
  kernel(n, m, alpha, dev_A, dev_B);
  cudaMemcpy(B, dev_B, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(m, n, B);

  /* Be clean. */
  free((void *)A);
  free((void *)B);

  return 0;
}
