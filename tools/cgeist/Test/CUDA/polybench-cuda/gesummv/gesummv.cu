// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 10
// clang-format on
/**
 * gesummv.c: This file is part of the PolyBench 3.0 test suite.
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

#define RUN 100

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}

__global__ void kernel_y(int n, double alpha, double beta, double *A, double *B,
                         double tmp[], double x[], double y[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    tmp[i] = 0;
    y[i] = 0;
    for (int j = 0; j < n; j++) {
      tmp[i] += A[i * n + j] * x[j];
      y[i] += B[i * n + j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}

/* Array initialization. */
static void init_array(int n, double *A, double *B, double *x) {
  int i, j;

  for (i = 0; i < n; i++) {
    x[i] = ((double)i) / n;
    for (j = 0; j < n; j++) {
      A[i * n + j] = ((double)i * j) / n;
      B[i * n + j] = ((double)i * j) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double *y)

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf ", y[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = atoi(argv[2]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(n * n * sizeof(double));
  double *B = (double *)malloc(n * n * sizeof(double));
  double *tmp = (double *)malloc(n * sizeof(double));
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));

  //__builtin_assume(n>0);
  //__builtin_assume(n<0x7FFFFFFE);
  /* Initialize array(s). */
  init_array(n, A, B, x);
  double alpha = 43532;
  double beta = 12313;

  double *dev_A;
  double *dev_B;
  double *dev_tmp;
  double *dev_x;
  double *dev_y;
  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMalloc(&dev_B, n * n * sizeof(double));
  cudaMalloc(&dev_tmp, n * sizeof(double));
  cudaMalloc(&dev_x, n * sizeof(double));
  cudaMalloc(&dev_y, n * sizeof(double));
  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_tmp, tmp, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  /* Run kernel. */

  const unsigned threadsPerBlock = 256;
  kernel_y<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(
      n, alpha, beta, dev_A, dev_B, dev_tmp, dev_x, dev_y);
  cudaMemcpy(y, dev_y, n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(n, y);

  /* Be clean. */
  free((void *)A);
  free((void *)B);
  free((void *)tmp);
  free((void *)x);
  free((void *)y);

  cudaFree((void *)dev_A);
  cudaFree((void *)dev_B);
  cudaFree((void *)dev_tmp);
  cudaFree((void *)dev_x);
  cudaFree((void *)dev_y);
  return 0;
}
