// clang-format off
// RUN: cgeist %s %stdinclude %cudaopts -O3 -o %s.execm && %s.execm 1 5 10
// clang-format on
/**
 * jacobi-1d-imper.c: This file is part of the PolyBench 3.0 test suite.
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

__global__ void kernel_stencil(int n, double A[], double B[]) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

  if (i < n - 1) {
    B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3;
  }
}

static void kernel(int tsteps, int n, double A[], double B[]) {
  const unsigned int threadsPerBlock = 256;

  for (int t = 1; t <= tsteps; t++) {
    kernel_stencil<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, A,
                                                                        B);
    kernel_stencil<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, B,
                                                                        A);
  }
}

/* Array initialization. */
static void init_array(int n, double *A, double *B) {
  int i;

  for (i = 0; i < n; i++) {
    A[i] = ((double)i + 2) / n;
    B[i] = ((double)i + 3) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, double *A)

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf ", A[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int n = atoi(argv[3]);
  int tsteps = atoi(argv[2]);
  int dump_code = atoi(argv[1]);

  /* Variable declaration/allocation. */
  double *A = (double *)malloc(n * sizeof(double));
  double *B = (double *)malloc(n * sizeof(double));

  init_array(n, A, B);

  double *dev_A;
  double *dev_B;
  cudaMalloc(&dev_A, n * sizeof(double));
  cudaMalloc(&dev_B, n * sizeof(double));
  cudaMemcpy(dev_A, A, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, n * sizeof(double), cudaMemcpyHostToDevice);

  /* Run kernel. */
  kernel(tsteps, n, dev_A, dev_B);
  cudaMemcpy(A, dev_A, n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  if (dump_code == 1)
    print_array(n, A);

  /* Be clean. */
  free((void *)A);
  free((void *)B);

  return 0;
}
