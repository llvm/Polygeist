// clang-format off
// XFAIL: *
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm
// clang-format on
/**
 * mvt.c: This file is part of the PolyBench 3.0 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#define RUN 200
#define N 15000
// #define N 40

/* Array initialization. */
static void init_array(int n, double *x1, double *x2, double *y_1, double *y_2,
                       double *A) {
  int i, j;

  for (i = 0; i < n; i++) {
    x1[i] = ((double)i) / n;
    x2[i] = ((double)i + 1) / n;
    y_1[i] = ((double)i + 3) / n;
    y_2[i] = ((double)i + 4) / n;
    for (j = 0; j < n; j++)
      A[i * n + j] = ((double)i * j) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double *x1, double *x2)

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf", x1[i]);
    fprintf(stderr, "%0.2lf", x2[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
}

__global__ void kernel_x1(int n, double *x1, double *x2, double *y_1,
                          double *y_2, double *A) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  if (i < n) {
    for (j = 0; j < n; j++)
      x1[i] += A[i * n + j] * y_1[j];
  }
}

__global__ void kernel_x2(int n, double *x1, double *x2, double *y_1,
                          double *y_2, double *A) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  if (i < n) {
    for (j = 0; j < n; j++)
      x2[i] += A[j * n + i] * y_2[j];
  }
}

short num_blocks(short num, short factor) {
  return (num + factor - 1) / factor;
}

void kernel(int n, double *x1, double *x2, double *y_1, double *y_2,
            double *A) {
  short threadsPerBlock = 256;

  kernel_x1<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(n, x1, x2, y_1,
                                                                 y_2, A);
  kernel_x2<<<threadsPerBlock, num_blocks(n, threadsPerBlock)>>>(n, x1, x2, y_1,
                                                                 y_2, A);
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int dump_code = atoi(argv[1]);
  int n = N;

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(N * N * sizeof(double));
  double *x1 = (double *)malloc(sizeof(double) * n);
  double *x2 = (double *)malloc(sizeof(double) * n);
  double *y_1 = (double *)malloc(sizeof(double) * n);
  double *y_2 = (double *)malloc(sizeof(double) * n);

  /* Initialize array(s). */
  init_array(n, x1, x2, y_1, y_2, A);

  double *dev_A;
  double *dev_x1;
  double *dev_x2;
  double *dev_y_1;
  double *dev_y_2;

  cudaMalloc(&dev_A, n * n * sizeof(double));
  cudaMalloc(&dev_x1, n * sizeof(double));
  cudaMalloc(&dev_x2, n * sizeof(double));
  cudaMalloc(&dev_y_1, n * sizeof(double));
  cudaMalloc(&dev_y_2, n * sizeof(double));

  cudaMemcpy(dev_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x1, x1, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x2, x2, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y_1, y_1, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y_2, y_2, n * sizeof(double), cudaMemcpyHostToDevice);

  /* Run kernel. */
  kernel(n, dev_x1, dev_x2, dev_y_1, dev_y_2, dev_A);

  cudaMemcpy(x1, dev_x1, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(x2, dev_x2, n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(n, x1, x2);

  /* Be clean. */
  free((void *)A);
  free((void *)x1);
  free((void *)x2);
  free((void *)y_1);
  free((void *)y_2);

  return 0;
}
